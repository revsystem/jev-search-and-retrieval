"""Command line entry point."""

from __future__ import annotations

import argparse
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

    sweep = sub.add_parser("sweep", help="1リクエストに載せる候補数と精度の関係を測る")
    sweep.add_argument("--data", default=str(DEFAULT_PATH))
    sweep.add_argument("--batches", default="1,5,10,25,100")
    sweep.add_argument("--queries", type=int, default=30)
    sweep.add_argument("--candidates", type=int, default=100)
    sweep.add_argument("--k", type=int, default=10)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--out", default="out/sweep.json")
    sweep.set_defaults(func=cmd_sweep)

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
