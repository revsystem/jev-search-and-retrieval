"""Every table in the article, rebuilt from the scores `judge` and `dilution` saved.

Nothing here calls an API. Each measure works on the flat pair rows
(query_id, doc_id, label, score) that `jev-rag judge` writes to
out/judge-<dataset>.json, so anyone holding those files can reproduce the
numbers, including the intervals, from the same seed.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from jev_rag.judge import judge_metrics, paired_bootstrap, pooled_pr_auc

Rows = list[dict[str, Any]]

DATASETS = {"miracl": "MIRACL", "jragbench": "J-RAGBench", "synthetic": "合成データ"}
BATCHES = [
    ("pool=10,batch=1", "同じ問題の 10 件", "1 件ずつ"),
    ("pool=10,batch=10", "同じ問題の 10 件", "10 件まとめて"),
    ("pool=100,batch=5", "混ぜた 100 件", "5 件ずつ"),
    ("pool=100,batch=10", "混ぜた 100 件", "10 件ずつ"),
    ("pool=100,batch=25", "混ぜた 100 件", "25 件ずつ"),
    ("pool=100,batch=50", "混ぜた 100 件", "50 件ずつ"),
    ("pool=100,batch=100", "混ぜた 100 件", "100 件まとめて"),
]


def _mixed(rows: Rows) -> dict[str, Rows]:
    """Questions with both a relevant and an irrelevant candidate; only these can be ranked."""
    groups: dict[str, Rows] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    return {q: g for q, g in groups.items() if 0 < sum(r["label"] for r in g) < len(g)}


def _query_pr_aucs(rows: Rows) -> dict[str, float]:
    return {
        q: float(average_precision_score([r["label"] for r in g], [r["score"] for r in g]))
        for q, g in _mixed(rows).items()
    }


def mean_query_pr_auc(rows: Rows) -> float | None:
    """PR-AUC of each question on its own, averaged. What a reranker needs."""
    values = list(_query_pr_aucs(rows).values())
    return float(np.mean(values)) if values else None


def precision_at_recall(rows: Rows, keep: float = 0.9) -> tuple[float, float]:
    """One threshold for every question, set so `keep` of the relevant documents survive.

    Pairs are ranked by score and the threshold is the score at which the
    running count of relevant documents first reaches `keep` of the total.
    Every document scoring at or above it is kept, ties included, so the kept
    share can land slightly above `keep`. Returns (precision, threshold).
    """
    scores = np.array([r["score"] for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows])
    order = np.argsort(-scores, kind="stable")
    reached = int(np.searchsorted(np.cumsum(labels[order]), keep * labels.sum()))
    threshold = scores[order][reached]
    kept = scores >= threshold
    return float(labels[kept].sum() / kept.sum()), float(threshold)


def top_k_hit(rows: Rows, k: int) -> float:
    """Share of questions with at least one relevant document in the top k."""
    groups = _mixed(rows).values()
    hits = [any(r["label"] for r in sorted(g, key=lambda r: -r["score"])[:k]) for g in groups]
    return float(np.mean(hits))


def random_top_k(rows: Rows, k: int) -> float:
    """The same share expected when every question's candidates are shuffled."""
    chances = []
    for g in _mixed(rows).values():
        n, relevant = len(g), sum(r["label"] for r in g)
        drawn = min(k, n)
        chances.append(1 - comb(n - relevant, drawn) / comb(n, drawn))
    return float(np.mean(chances))


def random_query_pr_auc(rows: Rows, draws: int = 100, seed: int = 0) -> float:
    """Per-question PR-AUC of random scores, averaged over `draws` shuffles."""
    rng = np.random.default_rng(seed)
    values = []
    for g in _mixed(rows).values():
        labels = [r["label"] for r in g]
        values.append(
            np.mean([average_precision_score(labels, rng.random(len(g))) for _ in range(draws)])
        )
    return float(np.mean(values))


def _interval_of_means(a: dict[str, float], b: dict[str, float], samples: int, seed: int):
    """Paired bootstrap of a mean over questions, from values computed once per question."""
    queries = sorted(set(a) & set(b))
    diffs = np.array([a[q] - b[q] for q in queries])
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(samples)]
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def _precision_only(rows: Rows) -> float:
    return precision_at_recall(rows)[0]


def _pairs(result: dict[str, Any], route: str) -> Rows | None:
    entry = result.get(route)
    return entry["scored"]["pairs"] if entry and "scored" in entry else None


def _diff(a: float, b: float, digits: int) -> str:
    # the shown difference is the difference of the shown values
    return f"{round(a, digits) - round(b, digits):+.{digits}f}".replace("-", "−")


def _interval(low: float, high: float, digits: int, scale: float = 1.0) -> str:
    return f"{low * scale:+.{digits}f}〜{high * scale:+.{digits}f}".replace("-", "−")


def _points(a: float, b: float, ci: dict[str, Any]) -> str:
    interval = _interval(ci["low"], ci["high"], 1, 100)
    return f"{_diff(a * 100, b * 100, 1)} ポイント（{interval}）"


def _num(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in body]


def _ranking_section(judges, samples, seed) -> list[str]:
    body = []
    for name, result in judges.items():
        jev, cohere = _pairs(result, "jev_pointwise"), _pairs(result, "cohere_rerank")
        if not jev or not cohere:
            continue
        a, b = _query_pr_aucs(jev), _query_pr_aucs(cohere)
        mean_a, mean_b = float(np.mean(list(a.values()))), float(np.mean(list(b.values())))
        low, high = _interval_of_means(a, b, samples, seed)
        embedding = _pairs(result, "embedding")
        body.append(
            [
                f"{DATASETS.get(name, name)}（{len(a)} 問）",
                _num(random_query_pr_auc(cohere, seed=seed), 2),
                _num(mean_query_pr_auc(embedding) if embedding else None, 3),
                _num(mean_b, 3),
                _num(mean_a, 3),
                f"{_diff(mean_a, mean_b, 3)}（{_interval(low, high, 3)}）",
            ]
        )
    header = [
        "データ",
        "でたらめ",
        "埋め込み",
        "Cohere Rerank",
        "Jev",
        "Jev − Cohere Rerank（95% 区間）",
    ]
    return [
        "### 1 つの質問に対する候補文書を並べ替える力（問題ごとの PR-AUC の平均）",
        "",
    ] + _table(header, body)


def _pooled_section(judges, samples, seed) -> list[str]:
    body = []
    for name, result in judges.items():
        jev, cohere = _pairs(result, "jev_pointwise"), _pairs(result, "cohere_rerank")
        if not jev or not cohere:
            continue
        a, b = pooled_pr_auc(jev), pooled_pr_auc(cohere)
        ci = paired_bootstrap(jev, cohere, samples=samples, seed=seed)
        cells = [
            pooled_pr_auc(p) if (p := _pairs(result, r)) else None for r in ("bm25", "embedding")
        ]
        body.append(
            [
                DATASETS.get(name, name),
                _num(float(np.mean([r["label"] for r in cohere])), 2),
                _num(cells[0], 2),
                _num(cells[1], 2),
                _num(b, 2),
                _num(a, 2),
                f"{_diff(a, b, 2)}（{_interval(ci['low'], ci['high'], 2)}）",
            ]
        )
    header = [
        "データ",
        "でたらめ",
        "キーワード一致",
        "埋め込み",
        "Cohere Rerank",
        "Jev",
        "Jev − Cohere Rerank（95% 区間）",
    ]
    title = "### 質問をまたいでスコアの意味がそろっているか（全問題まとめた PR-AUC）"
    return [title, ""] + _table(header, body)


def _top_k_section(judges) -> list[str]:
    body = []
    for name, result in judges.items():
        cohere, jev = _pairs(result, "cohere_rerank"), _pairs(result, "jev_pointwise")
        if not cohere or not jev:
            continue
        label = DATASETS.get(name, name)
        body.append(
            [
                label,
                "でたらめに並べた場合",
                f"{random_top_k(cohere, 1):.0%}",
                f"{random_top_k(cohere, 3):.0%}",
            ]
        )
        for route, rows in (("Cohere Rerank", cohere), ("Jev", jev)):
            body.append([label, route, f"{top_k_hit(rows, 1):.0%}", f"{top_k_hit(rows, 3):.0%}"])
    header = ["データ", "並べ方", "1 位が関連文書", "上位 3 件に関連文書を含む"]
    title = "### 上位に関連文書が来るか（候補文書がすべて関連文書の問題を除く）"
    return [title, ""] + _table(header, body)


def _threshold_section(judges, samples, seed) -> list[str]:
    body, cuts = [], []
    for name, result in judges.items():
        jev, cohere = _pairs(result, "jev_pointwise"), _pairs(result, "cohere_rerank")
        if not jev or not cohere:
            continue
        embedding = _pairs(result, "embedding")
        (pa, ta), (pb, tb) = precision_at_recall(jev), precision_at_recall(cohere)
        pe, te = precision_at_recall(embedding) if embedding else (None, None)
        ci = paired_bootstrap(jev, cohere, samples=samples, seed=seed, metric=_precision_only)
        label = DATASETS.get(name, name)
        body.append(
            [
                label,
                f"{np.mean([r['label'] for r in cohere]) * 100:.1f}%",
                "-" if pe is None else f"{pe * 100:.1f}%",
                f"{pb * 100:.1f}%",
                f"{pa * 100:.1f}%",
                _points(pa, pb, ci),
            ]
        )
        cuts.append([label, _num(te, 2), _num(tb, 2), _num(ta, 2)])
    header = [
        "データ",
        "でたらめ",
        "埋め込み",
        "Cohere Rerank",
        "Jev",
        "Jev − Cohere Rerank（95% 区間）",
    ]
    title = (
        "### 1 つのしきい値で全質問を切ったとき"
        "（関連文書の 9 割を残し、残った文書のうち関連する割合）"
    )
    return (
        [title, ""]
        + _table(header, body)
        + ["", "しきい値:", ""]
        + _table(["データ", "埋め込み", "Cohere Rerank", "Jev"], cuts)
    )


def _criteria_section(judges, samples, seed) -> list[str]:
    ranking, pooled, cut = [], [], []
    for name, result in judges.items():
        jev, plain = _pairs(result, "jev_pointwise"), _pairs(result, "jev_pointwise_plain")
        cohere = _pairs(result, "cohere_rerank")
        if not jev or not plain or not cohere:
            continue
        label = DATASETS.get(name, name)
        ranking.append([label] + [_num(mean_query_pr_auc(r), 3) for r in (cohere, plain, jev)])
        pooled.append([label] + [_num(pooled_pr_auc(r), 2) for r in (cohere, plain, jev)])
        pa, pp = precision_at_recall(jev)[0], precision_at_recall(plain)[0]
        ci = paired_bootstrap(jev, plain, samples=samples, seed=seed, metric=_precision_only)
        cut.append(
            [
                label,
                f"{pp * 100:.1f}%",
                f"{pa * 100:.1f}%",
                _points(pa, pp, ci),
            ]
        )
    if not ranking:
        return []
    three = ["データ", "Cohere Rerank", "Jev 基準なし", "Jev 基準あり"]
    return (
        [
            "### 判定基準のありなし（jev_pointwise_plain と jev_pointwise）",
            "",
            "並べ替える力（問題ごとの PR-AUC の平均）:",
            "",
        ]
        + _table(three, ranking)
        + ["", "質問をまたいだ評価（全問題まとめた PR-AUC）:", ""]
        + _table(three, pooled)
        + ["", "1 つのしきい値で切ったとき（関連文書の 9 割を残す）:", ""]
        + _table(["データ", "Jev 基準なし", "Jev 基準あり", "差（95% 区間）"], cut)
    )


DECIDER_COMPARISON = [
    ("cohere_rerank", "Cohere Rerank"),
    ("jev_crossencode", "Jev（1 件ずつ）"),
    ("jev_pointwise", "Jev（10 件ずつ）"),
    ("decider_single", "strands-decider（1 件ずつ）"),
    ("decider_single_plain", "strands-decider（1 件ずつ、判定基準なし）"),
    ("decider_single_en", "strands-decider（1 件ずつ、英語）"),
    ("decider_pointwise", "strands-decider（最大 10 件ずつ）"),
]


def _decider_section(judges, samples, seed) -> list[str]:
    body, notes = [], []
    for name, result in judges.items():
        if not any(route.startswith("decider") for route in result):
            continue
        label = DATASETS.get(name, name)
        questions = len({r["query_id"] for r in _pairs(result, "cohere_rerank") or []}) or None
        for route, title in DECIDER_COMPARISON:
            rows = _pairs(result, route)
            if not rows:
                continue
            scored = result[route]["scored"]
            probabilistic = route.startswith(("jev", "decider"))
            ece = judge_metrics(rows, probabilistic=True)["ece"] if probabilistic else None
            seconds = scored.get("seconds")
            per_question = seconds / questions if seconds and questions else None
            latencies = scored.get("latencies_ms")
            body.append(
                [
                    label,
                    f"{title} `{route}`",
                    _num(mean_query_pr_auc(rows), 3),
                    _num(pooled_pr_auc(rows), 2),
                    f"{precision_at_recall(rows)[0] * 100:.1f}%",
                    _num(ece, 3),
                    _num(per_question, 2),
                    _num(float(np.median(latencies)), 0) if latencies else "-",
                ]
            )
        jev, decider = _pairs(result, "jev_pointwise"), _pairs(result, "decider_single")
        if jev and decider:
            low, high = _interval_of_means(
                _query_pr_aucs(jev), _query_pr_aucs(decider), samples, seed
            )
            diff = _diff(mean_query_pr_auc(jev), mean_query_pr_auc(decider), 3)
            notes.append(
                f"{label}: Jev（10 件ずつ） − strands-decider（1 件ずつ）の問題ごとの PR-AUC "
                f"{diff}（{_interval(low, high, 3)}）"
            )
    if not body:
        return []
    header = [
        "データ",
        "経路",
        "問題ごとの PR-AUC",
        "全問まとめ PR-AUC",
        "9 割を残したときの関連の割合",
        "ECE",
        "1 問あたりの秒",
        "1 リクエストの処理 ms（中央値）",
    ]
    title = "### strands-decider との比較（strands-decider はローカルの GPU で 1 件ずつ順に送信）"
    return [title, ""] + _table(header, body) + ([""] + notes if notes else [])


def _dilution_section(dilution, samples, seed) -> list[str]:
    body = []
    for key, pool, batch in BATCHES:
        entry = dilution.get(key)
        if not entry:
            continue
        share = entry["distractors"]["share_at_half"]
        body.append(
            [
                pool,
                batch,
                _num(mean_query_pr_auc(entry["own_pairs"]), 2),
                _num(pooled_pr_auc(entry["own_pairs"]), 2),
                "該当なし" if share is None else f"{share:.0%}",
            ]
        )
    lines = ["### 1 リクエストに載せる件数（jev-rag dilution）", ""] + _table(
        [
            "候補文書の中身",
            "1 リクエストの件数",
            "並べ替え（問題ごと）",
            "全問まとめ",
            "混ぜた文書を「関連」と誤った割合",
        ],
        body,
    )
    together, alone = dilution.get("pool=10,batch=10"), dilution.get("pool=10,batch=1")
    if together and alone:
        a, b = _query_pr_aucs(together["own_pairs"]), _query_pr_aucs(alone["own_pairs"])
        low, high = _interval_of_means(a, b, samples, seed)
        ci = paired_bootstrap(together["own_pairs"], alone["own_pairs"], samples=samples, seed=seed)
        lines += [
            "",
            "10 件まとめて − 1 件ずつ: 並べ替え "
            f"{_diff(float(np.mean(list(a.values()))), float(np.mean(list(b.values()))), 3)}"
            f"（{_interval(low, high, 3)}）、全問まとめ "
            f"{_diff(pooled_pr_auc(together['own_pairs']), pooled_pr_auc(alone['own_pairs']), 3)}"
            f"（{_interval(ci['low'], ci['high'], 3)}）",
        ]
    return lines


def format_report(
    judges: dict[str, dict[str, Any]],
    dilution: dict[str, Any] | None,
    samples: int = 2000,
    seed: int = 0,
) -> str:
    """All article tables as Markdown. Datasets are shown in the order given."""
    sections = [
        _ranking_section(judges, samples, seed),
        _pooled_section(judges, samples, seed),
        _top_k_section(judges),
        _threshold_section(judges, samples, seed),
        _criteria_section(judges, samples, seed),
        _decider_section(judges, samples, seed),
    ]
    if dilution:
        sections.append(_dilution_section(dilution, samples, seed))
    return "\n\n".join("\n".join(s) for s in sections if s) + "\n"
