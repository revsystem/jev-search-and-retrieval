"""TypeSafe's published Search and retrieval use cases, mapped to this project.

The bullets are quoted from the "Search and retrieval" section of
https://docs.typesafe.ai/concepts/use-case-map, and the decision shapes from
the "Example task categories" table on the same page. Keeping the mapping in
code rather than only in the README means a rename or a deletion breaks a test
instead of quietly making the documentation wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The decision shapes the use-case map enumerates.
DECISION_SHAPES = {
    "Classification",
    "Detection",
    "Scoring",
    "Routing",
    "Search",
    "Retrieval",
    "Ranking",
    "Verification",
    "ML Feature Extraction",
    "Structured Data Extraction",
}


@dataclass(frozen=True)
class UseCase:
    key: str
    bullet: str
    shapes: list[str]
    module: str
    tests: str
    pipelines: list[str] = field(default_factory=list)
    entrypoint: str = ""
    note: str = ""


USE_CASES: list[UseCase] = [
    UseCase(
        key="replace_or_supplement",
        bullet=(
            "Replace or supplement embeddings in RAG pipelines with semantic search, "
            "scoring, and ranking."
        ),
        shapes=["Search", "Retrieval", "Ranking"],
        module="jev_rag.jev.retrieval",
        tests="tests/test_uc1_jev_retrieval.py",
        pipelines=["jev_only", "jev_hybrid"],
        note=(
            "jev_only は埋め込みを経路から外し、コーパスをNoulで走査する。"
            "jev_hybrid は埋め込みを再現率の段、Jevを適合率の段として融合する。"
        ),
    ),
    UseCase(
        key="score_relevance",
        bullet="Score query-to-candidate relevance.",
        shapes=["Scoring", "Ranking"],
        module="jev_rag.jev.rerank",
        tests="tests/test_uc2_relevance_scoring.py",
        pipelines=["jev_noul"],
        note="候補ごとにNoulを1問立て、返った確率そのもので並べ替える。",
    ),
    UseCase(
        key="pairwise_rerank",
        bullet="Rerank results with pairwise comparisons.",
        shapes=["Ranking"],
        module="jev_rag.jev.pairwise",
        tests="tests/test_uc3_pairwise.py",
        pipelines=["jev_pairwise"],
        note="絶対尺度を使わず、2候補のChoiceの勝率で順位を決める。",
    ),
    UseCase(
        key="cross_encode",
        bullet="Cross-encode queries and candidates for higher precision.",
        shapes=["Scoring", "Ranking"],
        module="jev_rag.jev.crossencode",
        tests="tests/test_uc4_cross_encode.py",
        pipelines=["jev_crossencode"],
        note="ペアごとにリクエストを分け、stateに他候補を入れない。",
    ),
    UseCase(
        key="select_context",
        bullet="Select useful context for downstream AI workflows.",
        shapes=["Retrieval"],
        module="jev_rag.jev.context",
        tests="tests/test_uc5_context_selection.py",
        entrypoint="jev-rag ask --context-budget",
        note="有用性と重複を判定し、予算内に貪欲に詰める。順位ではなく採否を決める段。",
    ),
]


def format_registry() -> str:
    header = ["公式ユースケース", "決定の型", "実装", "実行方法", "テスト"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for use_case in USE_CASES:
        run = ", ".join(use_case.pipelines) or use_case.entrypoint
        lines.append(
            f"| {use_case.bullet} | {', '.join(use_case.shapes)} | "
            f"`{use_case.module}` | `{run}` | `{use_case.tests}` |"
        )
    return "\n".join(lines)
