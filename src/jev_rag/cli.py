"""Command line entry point."""

from __future__ import annotations

import argparse
import hashlib
import sys

from jev_rag.dataset import DEFAULT_PATH, load_jqara, sample_queries
from jev_rag.rankers import ALL_RANKERS, build_rankers
from jev_rag.report import format_comparison, format_movers
from jev_rag.runner import evaluate_rankers, load, save
from jev_rag.settings_file import DEFAULT_ENV_FILE, load_env_file

ENV_FILE = DEFAULT_ENV_FILE
RESULTS_PATH = "out/results.json"
METRICS = ["ndcg@10", "mrr@10", "recall@10"]


def cmd_prepare(args: argparse.Namespace) -> int:
    from jev_rag.dataset import download

    path = download(args.out)
    queries = load_jqara(path)
    relevant = sum(q.total_relevant for q in queries)
    print(f"{path} を取得しました")
    candidates = sum(len(q.candidates) for q in queries)
    print(f"  {len(queries)} 問 / 候補 {candidates} 件 / 正解 {relevant} 件")
    print("  出典: JQaRA (hotchpotch) — question/answers は JAQKET 由来 CC-BY-SA-4.0、")
    print("        passage は Wikipedia の CC BY-SA 4.0 または GFDL")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from jev_rag.config import Settings

    settings = Settings()
    queries = sample_queries(
        load_jqara(args.data, max_candidates=args.candidates), args.queries, seed=args.seed
    )
    names = args.rankers.split(",")
    client = settings.jev.build_client() if any(n.startswith("jev") for n in names) else None
    rankers = build_rankers(names, client=client, settings=settings.bedrock)

    total_calls = len(queries) * args.candidates
    print(f"{len(queries)} 問 x 候補{args.candidates}件 で {len(names)} 経路を評価します")
    print(f"  cross-encode を含む場合、Jevへのリクエストは最大 {total_calls} 件")

    def progress(name, done, total):
        if done % 10 == 0 or done == total:
            print(f"\r  {name}: {done}/{total}", end="", flush=True)
            if done == total:
                print()

    results = evaluate_rankers(
        rankers,
        queries,
        k=args.k,
        progress=progress,
        on_result=lambda partial: save(partial, args.out),
    )
    results["_run"] = {
        "queries": len(queries),
        "candidates": args.candidates,
        "seed": args.seed,
        "k": args.k,
    }
    print(f"\n{save(results, args.out)} に保存しました\n")
    print(
        format_comparison(
            {k: v for k, v in results.items() if k != "_run"},
            baseline=args.baseline,
            metrics=METRICS,
            published=True,
        )
    )
    return 0


def _preflight(settings, names: list[str], needs_jev: bool) -> None:
    """Fail now rather than forty minutes in.

    A route needing AWS and a route needing Jev are checked before any
    measurement starts, because an expiring credential mid-run costs the whole
    run and reads as a route scoring zero.
    """
    import boto3

    from jev_rag.preflight import check_aws, check_jev

    if any(not n.startswith("jev") or n == "jev_hybrid" for n in names):
        identity = check_aws(boto3.client("sts", region_name=settings.bedrock.region))
        print(f"AWS: {identity}")
    if needs_jev:
        check_jev(settings.jev.build_client())
        print(f"Jev: {settings.jev.transport} 経路で応答あり")


def cmd_answer(args: argparse.Namespace) -> int:
    """End to end: does a better ranking change the answer the pipeline gives?

    Everything but the ranking route is held fixed — same questions, same
    candidate pool, same top-k, same prompt, same generation model — so a
    difference in the answer is attributable to the ranking.
    """
    from jev_rag.bedrock import BedrockGenerator
    from jev_rag.config import Settings
    from jev_rag.endtoend import answer_queries, format_answers
    from jev_rag.jev.context import ContextSelector

    settings = Settings()
    names = args.rankers.split(",")
    needs_jev = args.select or any(n.startswith("jev") for n in names)
    client = settings.jev.build_client() if needs_jev else None
    _preflight(settings, names, needs_jev)
    rankers = build_rankers(names, client=client, settings=settings.bedrock)
    generator = BedrockGenerator(settings.bedrock)
    queries = sample_queries(
        load_jqara(args.data, max_candidates=args.candidates), args.queries, seed=args.seed
    )

    if args.select:
        # the context-selection use case, applied after the ranking it follows
        selector = ContextSelector(client, budget_chars=args.budget)

        class Selected:
            def __init__(self, inner):
                self.inner = inner
                self.name = f"{inner.name}+select"

            def rank(self, question, docs):
                ranked = self.inner.rank(question, docs)[: args.top_k]
                return selector.select(question, ranked).selected

        rankers = {r.name: r for r in (Selected(v) for v in rankers.values())}
        baseline = f"{args.baseline}+select"
    else:
        baseline = args.baseline

    def progress(name, done, total):
        if done % 10 == 0 or done == total:
            print(f"\r  {name}: {done}/{total}", end="", flush=True)
            if done == total:
                print()

    results: dict = {}
    for name, ranker in rankers.items():
        try:
            results[name] = answer_queries(
                ranker, queries, generator, top_k=args.top_k, progress=progress
            )
        except Exception as error:  # noqa: BLE001 - reported per route, run continues
            results[name] = {"error": f"{type(error).__name__}: {error}"}
        save(results, args.out)

    from jev_rag.bedrock import SYSTEM_PROMPT

    results["_run"] = {
        "queries": len(queries),
        "candidates": args.candidates,
        "top_k": args.top_k,
        "seed": args.seed,
        "select": args.select,
        "budget": args.budget if args.select else None,
        "generation_model": settings.bedrock.generation_model_id,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16],
    }
    save(results, args.out)
    print(f"\n{args.out} に保存しました\n")
    print(f"{len(queries)} 問、候補{args.candidates}件から上位{args.top_k}件を文脈に渡して回答")
    print(format_answers(results, baseline=baseline))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """How accuracy moves with the number of candidates sharing one request.

    TypeSafe documents accuracy falling as the state fills with detail
    unrelated to the decision. In a reranking request that detail is the other
    candidates, so this sweeps the batch size from cross-encoding (1) up to the
    whole shortlist and reports the curve.
    """
    from jev_rag.config import Settings
    from jev_rag.rankers import JevPointwiseRanker

    settings = Settings()
    client = settings.jev.build_client()
    queries = sample_queries(
        load_jqara(args.data, max_candidates=args.candidates), args.queries, seed=args.seed
    )
    sizes = [int(size) for size in args.batches.split(",")]
    rankers = {f"batch={size}": JevPointwiseRanker(client, batch_size=size) for size in sizes}

    def progress(name, done, total):
        if done % 10 == 0 or done == total:
            print(f"\r  {name}: {done}/{total}", end="", flush=True)
            if done == total:
                print()

    results = evaluate_rankers(
        rankers,
        queries,
        k=args.k,
        progress=progress,
        on_result=lambda partial: save(partial, args.out),
    )
    print(f"\n{save(results, args.out)} に保存しました\n")
    print(
        f"候補{args.candidates}件を何件ずつ1リクエストに載せるかの比較"
        "（batch=1 はクロスエンコード）"
    )
    print(format_comparison(results, baseline=f"batch={sizes[0]}", metrics=METRICS))
    return 0


def _judge_queries(dataset: str):
    if dataset == "synthetic":
        from jev_rag.synthetic import load_synthetic

        return [q for part in ("gen_a", "gen_b") for q in load_synthetic(f"data/synthetic/{part}")]
    from jev_rag.public import load_public

    return load_public()[dataset]


def cmd_judge(args: argparse.Namespace) -> int:
    """Each route as a relevance judge: one score per (query, document) pair."""
    from jev_rag.config import Settings
    from jev_rag.judge import format_judge, run_judge

    settings = Settings()
    queries = sample_queries(_judge_queries(args.dataset), args.queries, seed=args.seed)
    names = [n for n in args.rankers.split(",") if n]
    needs_jev = any(n.startswith("jev") for n in names)
    if names:
        _preflight(settings, names, needs_jev=needs_jev)
    client = settings.jev.build_client() if needs_jev else None
    rankers = build_rankers(names, client=client, settings=settings.bedrock)
    pairs = sum(len(q.candidates) for q in queries)
    print(
        f"{args.dataset}: {len(queries)} 問 / ペア {pairs} 件、{len(names)} 経路を判定器として評価"
    )
    out = args.out or f"out/judge-{args.dataset}.json"
    results = run_judge(rankers, queries, out)
    print(f"{out} に保存しました\n")
    print(format_judge(results))
    return 0


def cmd_dilution(args: argparse.Namespace) -> int:
    """How many documents can share one request before judging suffers.

    Each synthetic question's own ten documents are scored while documents
    written for other questions pad the pool, so the request carries more and
    more detail unrelated to the decision.
    """
    from jev_rag.config import Settings
    from jev_rag.judge import run_dilution
    from jev_rag.rankers import JevPointwiseRanker
    from jev_rag.synthetic import with_distractors

    settings = Settings()
    _preflight(settings, ["jev_pointwise"], needs_jev=True)
    client = settings.jev.build_client()
    queries = _judge_queries("synthetic")
    settings_list = [tuple(int(n) for n in item.split(":")) for item in args.settings.split(",")]
    pools = {
        (pool, batch): with_distractors(queries, pool, seed=args.seed)
        for pool, batch in settings_list
    }
    results = run_dilution(
        lambda batch: JevPointwiseRanker(client, batch_size=batch), pools, args.out
    )
    print(f"{args.out} に保存しました\n")
    print(
        "| 設定 | PR-AUC | 問題内ROC-AUC | ECE | 無関係の平均 | 無関係で0.5以上 | リクエスト | 秒 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, r in results.items():
        d = r["distractors"]
        mean = "-" if d["mean_score"] is None else f"{d['mean_score']:.3f}"
        share = "-" if d["share_at_half"] is None else f"{d['share_at_half']:.3f}"
        print(
            f"| {key} | {r['pr_auc']:.3f} | {r['per_query_roc_auc']:.3f} | {r['ece']:.3f} "
            f"| {mean} | {share} | {r['requests']} | {r['seconds']} |"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from jev_rag.runner import inconsistent_query_counts, merge_files

    paths = args.results.split(",")
    merged = merge_files(paths) if len(paths) > 1 else load(args.results)
    results = {k: v for k, v in merged.items() if k != "_run"}

    mismatched = inconsistent_query_counts(results)
    if mismatched:
        print("注意: 経路によって問題数が違う。同じ土俵の比較になっていない。")
        for name, count in sorted(mismatched.items()):
            print(f"  {name}: {count} 問")
        print()
    print(format_comparison(results, baseline=args.baseline, metrics=METRICS, published=True))
    if args.movers and args.movers in results:
        print()
        print(format_movers(results, baseline=args.baseline, ranker=args.movers, limit=args.limit))
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """Use case: select useful context. Reports what the selection costs and saves.

    The candidates are ranked first. Selecting from the raw shortlist would
    measure the selector against a near-random slice and count evidence a real
    pipeline would never have handed it.
    """
    from jev_rag.config import Settings
    from jev_rag.jev.context import ContextSelector
    from jev_rag.rankers import JevPointwiseRanker

    settings = Settings()
    client = settings.jev.build_client()
    queries = sample_queries(
        load_jqara(args.data, max_candidates=args.candidates), args.queries, seed=args.seed
    )
    ranker = JevPointwiseRanker(client)
    selector = ContextSelector(client, budget_chars=args.budget)

    kept_chars = total_chars = kept_relevant = total_relevant = kept_docs = 0
    answerable = still_answerable = 0
    for query in queries:
        docs = ranker.rank(query.question, query.as_documents())[: args.top_k]
        relevant = sum(1 for d in docs if d.label)
        if relevant:
            answerable += 1
        selection = selector.select(query.question, docs)
        total_chars += sum(len(d.text) for d in docs)
        kept_chars += selection.used_chars
        kept_docs += len(selection.selected)
        total_relevant += relevant
        survivors = sum(1 for d in selection.selected if d.label)
        kept_relevant += survivors
        # what a RAG prompt actually needs is one good passage, not all of them
        if relevant and survivors:
            still_answerable += 1

    print(f"{len(queries)} 問、並べ替え後の上位{args.top_k}件を入力、予算{args.budget}文字")
    saved = kept_chars / max(total_chars, 1)
    print(f"  渡す文脈: {total_chars} -> {kept_chars} 文字 ({saved:.0%})")
    per_query = kept_docs / max(len(queries), 1)
    print(f"  採用件数: {kept_docs} / {len(queries) * args.top_k}（1問あたり {per_query:.1f} 件）")
    retained = kept_relevant / max(total_relevant, 1)
    print(f"  正解の保持: {kept_relevant} / {total_relevant} ({retained:.0%})")
    print(f"  正解を1件以上残せた問題: {still_answerable} / {answerable}")
    print(f"  （上位{args.top_k}件に正解を含む問題が {answerable} / {len(queries)}）")
    return 0


def cmd_usecases(args: argparse.Namespace) -> int:
    from jev_rag.usecases import USE_CASES, format_registry

    print("TypeSafe 公式ドキュメント Search and retrieval の実装対応\n")
    print(format_registry())
    if args.verbose:
        print()
        for use_case in USE_CASES:
            print(f"- {use_case.bullet}\n  {use_case.note}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from jev_rag.config import Settings
    from jev_rag.jev.questions import Noul

    settings = Settings()
    if args.transport:
        settings.jev.transport = args.transport
    client = settings.jev.build_client()
    print(f"transport={settings.jev.transport} model={client.transport.model}")
    # exercises the documented backticked indexed path, so a passing check
    # confirms the reference syntax the whole comparison depends on
    answer = client.evaluate(
        {
            "query": "赤い惑星と呼ばれるのはどれか",
            "candidates": [
                {"text": "金星は分厚い大気に覆われている。"},
                {"text": "火星は赤い惑星と呼ばれる。"},
            ],
        },
        {
            "wrong": Noul("`candidates[0].text` は `query` に答える根拠になりますか。"),
            "right": Noul("`candidates[1].text` は `query` に答える根拠になりますか。"),
        },
    )
    print(f"noul(無関係)={answer['wrong'].value}  noul(正解)={answer['right'].value}")
    if answer["right"].value <= answer["wrong"].value:
        print("警告: 正解側が高くならなかった。state パスの参照が効いていない可能性がある。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-rag", description="従来RAG/rerank と Jev の比較")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="評価データセット JQaRA を取得する")
    prepare.add_argument("--out", default=str(DEFAULT_PATH))
    prepare.set_defaults(func=cmd_prepare)

    evaluate = sub.add_parser("evaluate", help="各経路を同じ問題で評価して比較表を出す")
    evaluate.add_argument("--data", default=str(DEFAULT_PATH))
    evaluate.add_argument("--rankers", default=",".join(ALL_RANKERS))
    evaluate.add_argument("--queries", type=int, default=100, help="評価する問題数")
    evaluate.add_argument("--candidates", type=int, default=30, help="1問あたりの候補数")
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--baseline", default="embedding")
    evaluate.add_argument("--out", default=RESULTS_PATH)
    evaluate.set_defaults(func=cmd_evaluate)

    answer = sub.add_parser("answer", help="検索経路を替えて最終回答の正解率を比べる")
    answer.add_argument("--data", default=str(DEFAULT_PATH))
    answer.add_argument("--rankers", default="embedding,cohere_rerank,jev_pointwise")
    answer.add_argument("--queries", type=int, default=100)
    answer.add_argument("--candidates", type=int, default=100)
    answer.add_argument("--top-k", type=int, default=5, help="生成に渡す文書数")
    answer.add_argument("--select", action="store_true", help="Jevの文脈選択を後段に挟む")
    answer.add_argument("--budget", type=int, default=1500)
    answer.add_argument("--seed", type=int, default=0)
    answer.add_argument("--baseline", default="embedding")
    answer.add_argument("--out", default="out/answers.json")
    answer.set_defaults(func=cmd_answer)

    sweep = sub.add_parser("sweep", help="1リクエストに載せる候補数と精度の関係を測る")
    sweep.add_argument("--data", default=str(DEFAULT_PATH))
    sweep.add_argument("--batches", default="1,5,10,25,100")
    sweep.add_argument("--queries", type=int, default=30)
    sweep.add_argument("--candidates", type=int, default=100)
    sweep.add_argument("--k", type=int, default=10)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--out", default="out/sweep.json")
    sweep.set_defaults(func=cmd_sweep)

    judge = sub.add_parser("judge", help="各経路を関連度の判定器として評価する")
    judge.add_argument("--dataset", choices=["synthetic", "jragbench", "miracl"], required=True)
    judge.add_argument("--rankers", default="embedding,cohere_rerank,jev_pointwise,jev_crossencode")
    judge.add_argument("--queries", type=int, default=10_000, help="評価する問題数（既定は全問）")
    judge.add_argument("--seed", type=int, default=0)
    judge.add_argument("--out", default=None, help="既定は out/judge-<dataset>.json")
    judge.set_defaults(func=cmd_judge)

    dilution = sub.add_parser(
        "dilution", help="無関係な文書を混ぜ、1リクエストの件数と判定精度を測る"
    )
    dilution.add_argument(
        "--settings",
        default="10:1,10:10,100:5,100:10,100:25,100:50,100:100",
        help="候補数:1リクエストの件数 をカンマ区切りで",
    )
    dilution.add_argument("--seed", type=int, default=0)
    dilution.add_argument("--out", default="out/dilution.json")
    dilution.set_defaults(func=cmd_dilution)

    report = sub.add_parser("report", help="保存済みの結果から表を出し直す")
    report.add_argument(
        "--results", default=RESULTS_PATH, help="カンマ区切りで複数指定すると統合して表示する"
    )
    report.add_argument("--baseline", default="embedding")
    report.add_argument("--movers", default="jev_crossencode", help="順位が上がった例を出す経路")
    report.add_argument("--limit", type=int, default=10)
    report.set_defaults(func=cmd_report)

    context = sub.add_parser("context", help="文脈選択が削る量と、残る正解の割合を測る")
    context.add_argument("--data", default=str(DEFAULT_PATH))
    context.add_argument("--queries", type=int, default=50)
    context.add_argument("--candidates", type=int, default=30)
    context.add_argument("--top-k", type=int, default=10)
    context.add_argument("--budget", type=int, default=1500)
    context.add_argument("--seed", type=int, default=0)
    context.set_defaults(func=cmd_context)

    usecases = sub.add_parser("usecases", help="公式ユースケースと実装の対応を表示する")
    usecases.add_argument("--verbose", action="store_true")
    usecases.set_defaults(func=cmd_usecases)

    check = sub.add_parser("check", help="Jevの疎通確認")
    check.add_argument("--transport", choices=["gateway", "direct", "fake"])
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    # before any command builds Settings, which reads os.environ
    load_env_file(ENV_FILE)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
