"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jev_rag.config import Settings
from jev_rag.documents import Chunk, chunk_pages, download_pdf, load_pdf
from jev_rag.evaluation import (
    build_qrels,
    compare,
    format_table,
    load_queries,
    mean_metrics,
    score_ranking,
)

CHUNK_CACHE = Path("data/cache/chunks.jsonl")


def _save_chunks(chunks: list[Chunk], enriched_metadata: dict[str, dict] | None = None) -> None:
    CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNK_CACHE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
            }
            if enriched_metadata:
                record["metadata"] = enriched_metadata.get(chunk.chunk_id, {})
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_chunks() -> list[Chunk]:
    if not CHUNK_CACHE.exists():
        raise SystemExit(f"{CHUNK_CACHE} がありません。先に `jev-rag ingest` を実行してください。")
    return [
        Chunk(chunk_id=r["chunk_id"], page=r["page"], section=r["section"], text=r["text"])
        for r in (json.loads(line) for line in CHUNK_CACHE.read_text(encoding="utf-8").splitlines())
    ]


def _build_retriever(settings: Settings):
    from jev_rag.bedrock import CohereEmbedder
    from jev_rag.vector_store import EmbeddingRetriever, S3VectorStore

    return EmbeddingRetriever(CohereEmbedder(settings.bedrock), S3VectorStore(settings.store))


def _build_pipelines(names: list[str], settings: Settings) -> dict:
    from jev_rag.bedrock import BedrockReranker
    from jev_rag.pipelines import BaselinePipeline, ClassicRerankPipeline, JevPipeline

    retriever = _build_retriever(settings)
    client = settings.jev.build_client() if any(n.startswith("jev") for n in names) else None

    available = {
        "baseline": lambda: BaselinePipeline(retriever, top_k=settings.top_k),
        "classic_rerank": lambda: ClassicRerankPipeline(
            retriever,
            BedrockReranker(settings.bedrock),
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
        ),
        "jev_rerank": lambda: JevPipeline(
            retriever, client, top_k=settings.top_k, candidate_k=settings.candidate_k
        ),
        "jev_full": lambda: JevPipeline(
            retriever,
            client,
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
            plan_query=True,
        ),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise SystemExit(f"未知のパイプライン: {sorted(unknown)}")
    return {name: available[name]() for name in names}


def cmd_demo(args: argparse.Namespace) -> int:
    from jev_rag.demo import run

    print(run(top_k=args.top_k))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings()
    path = Path(args.pdf) if args.pdf else download_pdf(settings.pdf_url, "data/raw/source.pdf")
    chunks = chunk_pages(load_pdf(path), settings.chunk_size, settings.chunk_overlap)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"{path} -> {len(chunks)} チャンク")

    texts = [chunk.embedding_text() for chunk in chunks]
    metadata = [{**chunk.metadata(), "text": chunk.text} for chunk in chunks]
    enriched_metadata: dict[str, dict] = {}

    if args.enrich:
        from jev_rag.jev.enrich import ChunkEnricher

        enriched = ChunkEnricher(settings.jev.build_client()).enrich(chunks)
        texts = [item.embedding_text() for item in enriched]
        metadata = [{**item.metadata(), "text": item.chunk.text} for item in enriched]
        enriched_metadata = {item.chunk.chunk_id: item.metadata() for item in enriched}
        labelled = sum(1 for item in enriched if item.topic)
        print(f"Jev enrichment: {labelled}/{len(enriched)} 件に分野ラベルを付与")

    _save_chunks(chunks, enriched_metadata)
    if args.dry_run:
        print("--dry-run: 埋め込みとS3 Vectorsへの書き込みはスキップしました")
        return 0

    from jev_rag.bedrock import CohereEmbedder
    from jev_rag.vector_store import S3VectorStore

    store = S3VectorStore(settings.store)
    store.ensure_index(settings.bedrock.embedding_dimension)
    vectors = CohereEmbedder(settings.bedrock).embed_documents(texts)
    store.put([chunk.chunk_id for chunk in chunks], vectors, metadata)
    print(f"{len(vectors)} 件を {settings.store.bucket}/{settings.store.index} に格納しました")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.top_k = args.top_k
    corpus = _load_chunks()
    queries = load_queries(args.queries)
    pipelines = _build_pipelines(args.pipelines.split(","), settings)

    results = {}
    for name, pipeline in pipelines.items():
        per_query = []
        for query in queries:
            ranked = pipeline.retrieve(query.question)
            per_query.append(score_ranking(ranked, build_qrels(query, corpus), k=args.top_k))
        results[name] = mean_metrics(per_query)

    metrics = [
        f"ndcg@{args.top_k}",
        f"strict_ndcg@{args.top_k}",
        f"recall@{args.top_k}",
        f"precision@{args.top_k}",
        "mrr",
    ]
    print(format_table(compare(results, baseline=args.baseline), metrics))
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    """Report how many chunks each keyword rule matches, to calibrate the eval set."""
    corpus = _load_chunks()
    print(f"corpus: {len(corpus)} チャンク\n")
    for query in load_queries(args.queries):
        qrels = build_qrels(query, corpus)
        grades = sorted({grade for grade in qrels.values()}, reverse=True)
        counts = ", ".join(f"grade{g}={sum(1 for v in qrels.values() if v == g)}" for g in grades)
        flag = "  <-- 要調整" if not qrels or max(qrels.values(), default=0) < 2 else ""
        print(f"{query.query_id}  {counts or '該当なし'}{flag}  {query.question}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from jev_rag.bedrock import BedrockGenerator

    settings = Settings()
    pipeline = _build_pipelines([args.pipeline], settings)[args.pipeline]
    docs = pipeline.retrieve(args.question)
    for doc in docs:
        print(
            f"[{doc.chunk_id}] score={doc.score:.3f} jev={doc.jev_score} Δrank={doc.rank_delta:+d}"
        )
    print("\n" + BedrockGenerator(settings.bedrock).answer(args.question, docs))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Send one trivial question to confirm the configured Jev route answers."""
    from jev_rag.jev.questions import Noul

    settings = Settings()
    if args.transport:
        settings.jev.transport = args.transport
    client = settings.jev.build_client()
    print(f"transport={settings.jev.transport} model={client.transport.model}")
    answer = client.evaluate(
        {"query": "火星について", "document": "火星は赤い惑星と呼ばれる。"},
        {"relevant": Noul("state.document は state.query の根拠になりますか。")},
    )
    print(f"noul={answer['relevant'].value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-rag", description="従来RAG/rerank と Jev の比較")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="認証情報なしで比較ハーネスを動かす")
    demo.add_argument("--top-k", type=int, default=5)
    demo.set_defaults(func=cmd_demo)

    ingest = sub.add_parser("ingest", help="PDFを取り込みS3 Vectorsへ格納する")
    ingest.add_argument("--pdf", help="ローカルPDFのパス（省略時は白書URLを取得）")
    ingest.add_argument("--enrich", action="store_true", help="Jevで索引時エンリッチを行う")
    ingest.add_argument("--limit", type=int, help="先頭N チャンクだけ処理する")
    ingest.add_argument("--dry-run", action="store_true", help="チャンク分割までで止める")
    ingest.set_defaults(func=cmd_ingest)

    evaluate = sub.add_parser("evaluate", help="各パイプラインを評価して比較表を出す")
    evaluate.add_argument("--queries", default="data/eval/queries.yaml")
    evaluate.add_argument("--pipelines", default="baseline,classic_rerank,jev_rerank,jev_full")
    evaluate.add_argument("--baseline", default="baseline")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.set_defaults(func=cmd_evaluate)

    labels = sub.add_parser("labels", help="キーワードルールのマッチ件数を確認する")
    labels.add_argument("--queries", default="data/eval/queries.yaml")
    labels.set_defaults(func=cmd_labels)

    ask = sub.add_parser("ask", help="検索して GPT-5.6 Luna で回答を生成する")
    ask.add_argument("question")
    ask.add_argument("--pipeline", default="jev_rerank")
    ask.set_defaults(func=cmd_ask)

    check = sub.add_parser("check", help="Jevの疎通確認")
    check.add_argument("--transport", choices=["gateway", "direct", "fake"])
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
