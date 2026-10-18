# OpenAI Decisions API と Jev 1.14 の判定器比較 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decisions API（`gpt-6-luna`）の判定経路 3 つを `jev-rag judge` に足し、Jev 1.14 の 3 経路と同じデータ・指示文・指標で比べて、`report-judge` に比較の表を出す。

**Architecture:** Decisions の呼び出しと経路は新しいモジュール `src/jev_rag/decisions.py` に置き、`rankers.build_rankers` に登録する（strands-decider の比較と同じ組み込み方）。両 API の「答えたモデル、請求トークン、拒否・欠けた答え、リクエストの秒数」は共通の `UsageMeter` に記録し、`run_judge` が経路ごとに `scored["usage"]` として保存する。順序の入れ替え確認は `src/jev_rag/probe.py` と `jev-rag probe-order` で行う。

**Tech Stack:** Python 3.11+、uv、openai（Python SDK）3.26.0 以上、pytest、numpy、scipy（scikit-learn の依存として入っている）、scikit-learn

**Spec:** `.claude/docs/specs/openai-decisions-comparison.md`

## Global Constraints

- Decisions のモデルは `gpt-6-luna`（現在これだけ）。エンドポイントは SDK の `client.decisions.create`
- `openai>=3.26.0`（Decisions に対応した最初のバージョン）。追加後の `uv.lock` は同じ commit に含める
- Jev は `JEV_DIRECT_MODEL=jev-1.14.0` で測る。`jev-latest` は 2026-10-19 時点で `jev-1.13.0` を指す
- 指示文は `src/jev_rag/jev/prompts.py` と `src/jev_rag/jev/crossencode.py` の文字列を変えずに使う
- 判定基準は `instructions` の末尾に `\n判定基準: はい（true）= <true の定義>いいえ（false）= <false の定義>` の形で足す（定義の文は `RELEVANCE_CRITERIA`）
- まとめる経路の件数は 10、並列度は 8（Jev と同じ）
- 料金（2026-10-19 の公式ページ）: Decisions 入力 $0.10/1M トークン、Jev 1.14 入力 $0.042/1M トークン
- 出力先は `out/decisions/`。`results/` と既存の `out/judge-*.json` は変えない
- 順序の入れ替え確認の基準: 差の絶対値の中央値 0.1 以下、かつスピアマン 0.8 以上。まとめる経路の問題ごと PR-AUC の平均が 1 件ずつの経路より 0.2 以上低ければパスを解釈できていないとみなす
- untracked の `CLAUDE.md` は commit しない。`git add` はファイルを列挙して行う
- テストは `uv run pytest -q`、lint は `uv run ruff check src tests` と `uv run ruff format --check src tests`
- ruff は E501（100 文字）を検査する。`ruff format` は文字列を分けないので、計画のコードで 100 文字を超える行は、括弧の中で文字列を連結するか引数ごとに改行して収める
- commit メッセージは英語のコンベンショナルコミット。末尾に `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` を付ける

---

## ファイル構成

| パス | 役割 | 新規/変更 |
|---|---|---|
| `src/jev_rag/usage.py` | 経路ごとの利用量（モデル、トークン、拒否、欠け、秒数）を数える `UsageMeter` | 新規 |
| `src/jev_rag/jev/client.py` | `JevClient` に `meter` を持たせ、応答ごとに記録する | 変更 |
| `src/jev_rag/rankers.py` | Jev の経路に `meter` を出し、Decisions の経路を登録する | 変更 |
| `src/jev_rag/judge.py` | `run_judge` が `usage` を保存し、Decisions を確率として扱う。`format_judge` に拒否と欠けの列 | 変更 |
| `src/jev_rag/decisions.py` | Decisions のクライアントと 3 経路 | 新規 |
| `src/jev_rag/config.py` | `DecisionsSettings` | 変更 |
| `src/jev_rag/preflight.py` | `check_decisions` | 変更 |
| `src/jev_rag/cli.py` | `_needs_aws`、Decisions の preflight、`judge` の配線、`probe-order` | 変更 |
| `src/jev_rag/probe.py` | 順序の入れ替え確認 | 新規 |
| `src/jev_rag/judge_report.py` | 「Decisions と Jev 1.14」の節 | 変更 |
| `tests/test_usage.py`、`tests/test_decisions.py`、`tests/test_probe.py` | 新規のテスト | 新規 |
| `tests/test_jev_client.py`、`tests/test_judge.py`、`tests/test_preflight.py`、`tests/test_rankers.py`、`tests/test_judge_report.py` | 既存のテストに追加 | 変更 |
| `.claude/docs/research/decisions-probe.md`、`.claude/docs/research/decisions-results.md` | 確認と本測定の結果 | 新規 |

---

### Task 1: UsageMeter と、Jev の応答の記録

**Files:**
- Create: `src/jev_rag/usage.py`
- Modify: `src/jev_rag/jev/client.py`（`__init__` と `_send`）
- Modify: `src/jev_rag/rankers.py`（`JevPointwiseRanker`、`JevCrossEncodeRanker` に `meter` プロパティ）
- Test: `tests/test_usage.py`（新規）、`tests/test_jev_client.py`（追加）

**Interfaces:**
- Produces: `UsageMeter.record(*, model: str, input_tokens: int, seconds: float, refusals: int = 0, missing: int = 0) -> None`、`UsageMeter.reset() -> None`、`UsageMeter.snapshot() -> dict`（キーは `requests`、`input_tokens`、`refusals`、`missing`、`models`（ソート済みのリスト）、`request_seconds`（リスト））
- Produces: `JevClient.meter: UsageMeter`、`JevPointwiseRanker.meter`、`JevCrossEncodeRanker.meter`（どちらも内側の `JevClient.meter`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_usage.py`:

```python
"""Per-route usage: the model that answered, the tokens billed, and answers that never came."""

from concurrent.futures import ThreadPoolExecutor

from jev_rag.usage import UsageMeter


def test_the_meter_adds_up_every_request():
    meter = UsageMeter()
    meter.record(model="m-1", input_tokens=100, seconds=0.5, refusals=1)
    meter.record(model="m-1", input_tokens=50, seconds=0.25, missing=2)
    assert meter.snapshot() == {
        "requests": 2,
        "input_tokens": 150,
        "refusals": 1,
        "missing": 2,
        "models": ["m-1"],
        "request_seconds": [0.5, 0.25],
    }


def test_reset_starts_a_new_route_from_zero():
    meter = UsageMeter()
    meter.record(model="m-1", input_tokens=100, seconds=0.5)
    meter.reset()
    assert meter.snapshot()["requests"] == 0
    assert meter.snapshot()["models"] == []


def test_requests_from_worker_threads_are_all_counted():
    meter = UsageMeter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.record(model="m", input_tokens=1, seconds=0.0), range(400)))
    assert meter.snapshot()["input_tokens"] == 400
```

`tests/test_jev_client.py` の末尾に追加:

```python
def test_the_client_meters_the_model_tokens_and_missing_answers():
    from jev_rag.jev.questions import Noul

    class Partial(FakeTransport):
        def send(self, body):
            response = super().send(body)
            response["answers"].pop("b")
            response["usage"] = {"input_tokens": 42}
            return response

    client = JevClient(Partial(noul=0.7))
    client.evaluate({"q": "x"}, {"a": Noul("a?"), "b": Noul("b?")})
    usage = client.meter.snapshot()
    assert usage["models"] == ["jev-fake"]
    assert usage["input_tokens"] == 42
    assert usage["missing"] == 1
    assert usage["requests"] == 1
```

（`tests/test_jev_client.py` の先頭で `FakeTransport` と `JevClient` が import されていなければ、`from jev_rag.jev.client import JevClient` と `from jev_rag.jev.transport import FakeTransport` を足す。）

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_usage.py tests/test_jev_client.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_rag.usage'`）

- [ ] **Step 3: 最小の実装を書く**

`src/jev_rag/usage.py`:

```python
"""Per-route usage: which model answered, the tokens billed, and answers that never came.

A route's requests run on worker threads, so every update takes a lock.
``run_judge`` resets the meter before a route and saves a snapshot after it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageMeter:
    requests: int = 0
    input_tokens: int = 0
    refusals: int = 0
    missing: int = 0
    models: set[str] = field(default_factory=set)
    seconds: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        seconds: float,
        refusals: int = 0,
        missing: int = 0,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += input_tokens
            self.refusals += refusals
            self.missing += missing
            if model:
                self.models.add(model)
            self.seconds.append(seconds)

    def reset(self) -> None:
        with self._lock:
            self.requests = self.input_tokens = self.refusals = self.missing = 0
            self.models = set()
            self.seconds = []

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "refusals": self.refusals,
                "missing": self.missing,
                "models": sorted(self.models),
                "request_seconds": list(self.seconds),
            }
```

`src/jev_rag/jev/client.py` の変更。import に `import time` と `from jev_rag.usage import UsageMeter` を足し、`__init__` の末尾に `self.meter = UsageMeter()` を足し、`_send` を次に置き換える:

```python
    def _send(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        wire = {key: question.payload() for key, question in questions.items()}
        started = time.monotonic()
        response = self.transport.send(self.transport.build_body(state, wire))
        returned = response.get("answers") or {}
        self.meter.record(
            model=str(response.get("model") or ""),
            input_tokens=int((response.get("usage") or {}).get("input_tokens", 0)),
            seconds=time.monotonic() - started,
            missing=sum(1 for key in questions if key not in returned),
        )
        return response
```

`src/jev_rag/rankers.py` の `JevPointwiseRanker` に追加:

```python
    @property
    def meter(self):
        return self.reranker.client.meter
```

`JevCrossEncodeRanker` に追加:

```python
    @property
    def meter(self):
        return self.encoder.client.meter
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q`
Expected: すべて PASS（既存の 338 件に新しい 4 件）

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/usage.py src/jev_rag/jev/client.py src/jev_rag/rankers.py tests/test_usage.py tests/test_jev_client.py
git commit -m "feat: meter the answering model, billed tokens and missing answers per Jev request"
```

---

### Task 2: run_judge が利用量を保存し、拒否と欠けを表示する

**Files:**
- Modify: `src/jev_rag/judge.py`（`run_judge`、`format_judge`）
- Test: `tests/test_judge.py`（追加）

**Interfaces:**
- Consumes: Task 1 の `ranker.meter`（`UsageMeter`）。`meter` を持たない経路（埋め込みなど）はそのまま動く
- Produces: `results[name]["scored"]["usage"]`（`UsageMeter.snapshot()` の dict）。`decisions` で始まる経路も `probabilistic=True` で指標を計算する。`format_judge` の表に「拒否/欠け」の列

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_judge.py` の末尾に追加:

```python
class MeteredRanker:
    """Scores every candidate 0.9 and records one refusal per question."""

    def __init__(self, name):
        from jev_rag.usage import UsageMeter

        self.name = name
        self.meter = UsageMeter()

    def rank(self, question, docs):
        self.meter.record(model="gpt-6-luna", input_tokens=10, seconds=0.1, refusals=1)
        for doc in docs:
            doc.score = 0.9 if doc.label else 0.1
        return docs


def labelled_queries():
    return [
        EvalQuery(
            query_id=f"q{i}",
            question="問",
            candidates=[
                RetrievedDoc(doc_id=f"q{i}-a", text="a", score=0.0, label=1),
                RetrievedDoc(doc_id=f"q{i}-b", text="b", score=0.0, label=0),
            ],
        )
        for i in range(3)
    ]


def test_run_judge_saves_each_routes_usage_from_a_fresh_meter(tmp_path):
    ranker = MeteredRanker("decisions_pointwise")
    ranker.meter.record(model="stale", input_tokens=999, seconds=1.0)
    results = run_judge({"decisions_pointwise": ranker}, labelled_queries(), tmp_path / "j.json")
    usage = results["decisions_pointwise"]["scored"]["usage"]
    assert usage["models"] == ["gpt-6-luna"]
    assert usage["input_tokens"] == 30
    assert usage["refusals"] == 3


def test_decisions_routes_get_calibration_metrics(tmp_path):
    results = run_judge(
        {"decisions_pointwise": MeteredRanker("decisions_pointwise")},
        labelled_queries(),
        tmp_path / "j.json",
    )
    assert "ece" in results["decisions_pointwise"]


def test_the_judge_table_shows_refusals_and_missing_answers(tmp_path):
    results = run_judge(
        {"decisions_pointwise": MeteredRanker("decisions_pointwise")},
        labelled_queries(),
        tmp_path / "j.json",
    )
    table = format_judge(results)
    assert "拒否/欠け" in table
    assert "| 3/0 |" in table
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_judge.py -q`
Expected: FAIL（`KeyError: 'usage'`、`ece` が無い、「拒否/欠け」が無い）

- [ ] **Step 3: 最小の実装を書く**

`run_judge` の経路のループを次に置き換える:

```python
    for name, ranker in rankers.items():
        if "pr_auc" in results.get(name, {}):
            continue
        meter = getattr(ranker, "meter", None)
        if meter is not None:
            # a client is shared by routes, so each route starts from zero
            meter.reset()
        scored = score_pairs(ranker, queries)
        if meter is not None:
            scored["usage"] = meter.snapshot()
        if scored["pairs"]:
            # both APIs return P(true): Jev from a Noul, Decisions from a predicate
            metrics = judge_metrics(
                scored["pairs"], probabilistic=name.startswith(("jev", "decisions"))
            )
        else:
            metrics = {"error": "全問失敗"}
        results[name] = metrics | {"scored": scored}
        save(results, target)
```

`format_judge` の変更。`header` の `"ペア数"` の前に `"拒否/欠け"` を足し、`cells` の `str(result.get("pairs", "-"))` の前に次の要素を足す:

```python
            _answer_gaps(result),
```

`format_judge` の直前に関数を足す:

```python
def _answer_gaps(result: dict[str, Any]) -> str:
    """Refusals and answers that never came, or '-' for a route with no meter."""
    usage = result.get("scored", {}).get("usage")
    return f"{usage['refusals']}/{usage['missing']}" if usage else "-"
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q`
Expected: すべて PASS。既存の `format_judge` のテストが列数で落ちた場合は、そのテストの期待値に「拒否/欠け」の列を足す（列を足したことによる期待値の更新だけにとどめる）

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/judge.py tests/test_judge.py
git commit -m "feat: save each route's usage in judge results and show refusals and missing answers"
```

---

### Task 3: Decisions のクライアント

**Files:**
- Modify: `pyproject.toml`、`uv.lock`（`uv add` で更新）
- Create: `src/jev_rag/decisions.py`（クライアント部分）
- Test: `tests/test_decisions.py`（新規）

**Interfaces:**
- Consumes: Task 1 の `UsageMeter`
- Produces: `DECISIONS_MODEL = "gpt-6-luna"`、`with_criteria(instruction: str) -> str`、`predicate(name: str, instructions: str) -> dict`、`DecisionsClient(sdk, model=DECISIONS_MODEL)`、`DecisionsClient.evaluate(input_text: str, questions: list[dict]) -> dict[str, dict]`（答えを名前で引ける dict。拒否と欠けは含めず `meter` に数える）、`DecisionsClient.meter: UsageMeter`

- [ ] **Step 1: 依存を足す**

Run: `uv add 'openai>=3.26.0'`
Expected: `pyproject.toml` の `dependencies` に `openai>=3.26.0` が入り、`uv.lock` が更新される。`uv run python -c "import openai; print(openai.__version__)"` が 3.26.0 以上を出す

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_decisions.py`:

```python
"""OpenAI's Decisions API as a relevance judge, exercised against a scripted SDK."""

import json

from openai.types import Decision

from jev_rag.decisions import DecisionsClient, predicate, with_criteria


class FakeDecisions:
    """Stands in for `OpenAI().decisions`. `reply` builds the answers for one request."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def create(self, **body):
        self.requests.append(body)
        return Decision.model_validate(
            {
                "answers": self.reply(body),
                "model": "gpt-6-luna",
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 120,
                },
            }
        )


class FakeSDK:
    def __init__(self, reply):
        self.decisions = FakeDecisions(reply)


def predicates(probabilities):
    """Answer each predicate with the probability keyed by its name, in reverse order."""

    def reply(body):
        return [
            {"type": "predicate", "name": q["name"], "probability": probabilities[q["name"]]}
            for q in reversed(body["questions"])
        ]

    return reply


def test_criteria_are_appended_to_the_instruction_text():
    text = with_criteria("`candidate` は役に立ちますか。")
    assert text == (
        "`candidate` は役に立ちますか。\n判定基準: はい（true）= "
        "クエリが指す対象そのものを扱い、回答を裏づける事実・数値・定義を含む。"
        "いいえ（false）= 話題が重なるだけ、参照先が異なる、または根拠となる情報を含まない。"
    )


def test_the_client_sends_the_model_input_and_questions():
    sdk = FakeSDK(predicates({"c0": 0.8}))
    client = DecisionsClient(sdk)
    client.evaluate('{"query": "x"}', [predicate("c0", "質問")])
    assert sdk.decisions.requests == [
        {
            "model": "gpt-6-luna",
            "input": '{"query": "x"}',
            "questions": [{"type": "predicate", "name": "c0", "instructions": "質問"}],
        }
    ]


def test_answers_are_matched_by_name_not_by_order():
    client = DecisionsClient(FakeSDK(predicates({"c0": 0.8, "c1": 0.1})))
    answers = client.evaluate("{}", [predicate("c0", "a"), predicate("c1", "b")])
    assert answers["c0"]["probability"] == 0.8
    assert answers["c1"]["probability"] == 0.1


def test_refusals_and_absent_answers_are_left_out_and_counted():
    def reply(body):
        return [
            {"type": "refusal", "name": "c0"},
            {"type": "predicate", "name": "c1", "probability": 0.4},
        ]

    client = DecisionsClient(FakeSDK(reply))
    answers = client.evaluate("{}", [predicate(f"c{i}", "q") for i in range(3)])
    assert set(answers) == {"c1"}
    usage = client.meter.snapshot()
    assert usage["refusals"] == 1
    assert usage["missing"] == 1


def test_the_meter_records_the_model_and_billed_tokens():
    client = DecisionsClient(FakeSDK(predicates({"c0": 0.5})))
    client.evaluate("{}", [predicate("c0", "q")])
    usage = client.meter.snapshot()
    assert usage["models"] == ["gpt-6-luna"]
    assert usage["input_tokens"] == 120
    assert usage["requests"] == 1


def test_the_input_keeps_japanese_readable():
    sdk = FakeSDK(predicates({"c0": 0.5}))
    DecisionsClient(sdk).evaluate(json.dumps({"query": "火星"}, ensure_ascii=False), [predicate("c0", "q")])
    assert "火星" in sdk.decisions.requests[0]["input"]
```

- [ ] **Step 3: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_decisions.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_rag.decisions'`）

- [ ] **Step 4: 最小の実装を書く**

`src/jev_rag/decisions.py`:

```python
"""OpenAI's Decisions API as a relevance judge, for comparison with Jev.

The Decisions API (`POST /v1/decisions`, public beta, `gpt-6-luna` only)
answers typed questions about one shared input, as Jev does, but two parts of
Jev's request have no counterpart:

- There is no `criteria` field. The written definition of relevance is
  appended to the instruction text instead.
- The input is a string or messages, not structured state, and nothing in the
  documentation says a backticked path such as `candidates[3].text` is
  resolved. The state is sent as JSON text with the same paths; whether the
  model follows them is checked with `jev-rag probe-order` before a full run.

Answers are matched to questions by name. A refusal or an answer that never
came scores 0.0, as Jev's missing answers do, and is counted on the meter so
the report shows how many there were.
"""

from __future__ import annotations

import time
from typing import Any

from jev_rag.jev.prompts import RELEVANCE_CRITERIA
from jev_rag.usage import UsageMeter

DECISIONS_MODEL = "gpt-6-luna"


def with_criteria(instruction: str) -> str:
    """The instruction followed by the definition Jev receives in `criteria`."""
    return (
        f"{instruction}\n判定基準: はい（true）= {RELEVANCE_CRITERIA['true']}"
        f"いいえ（false）= {RELEVANCE_CRITERIA['false']}"
    )


def predicate(name: str, instructions: str) -> dict[str, Any]:
    return {"type": "predicate", "name": name, "instructions": instructions}


class DecisionsClient:
    def __init__(self, sdk: Any, model: str = DECISIONS_MODEL) -> None:
        self.sdk = sdk
        self.model = model
        self.meter = UsageMeter()

    def evaluate(self, input_text: str, questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Answers by question name; refusals and absent names are left out and counted."""
        started = time.monotonic()
        decision = self.sdk.decisions.create(
            model=self.model, input=input_text, questions=questions
        )
        answers: dict[str, dict[str, Any]] = {}
        refused: set[str | None] = set()
        for answer in decision.answers:
            body = answer.model_dump()
            if body["type"] == "refusal":
                refused.add(body.get("name"))
            else:
                answers[body["name"]] = body
        self.meter.record(
            model=decision.model,
            input_tokens=decision.usage.input_tokens,
            seconds=time.monotonic() - started,
            refusals=len(refused),
            missing=sum(1 for q in questions if q["name"] not in answers and q["name"] not in refused),
        )
        return answers
```

- [ ] **Step 5: テストが通ることを確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: すべて PASS。ruff が整形を求めたら `uv run ruff format src tests` を実行してから再確認する

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/jev_rag/decisions.py tests/test_decisions.py
git commit -m "feat: add a Decisions API client that matches answers by name and meters refusals"
```

---

### Task 4: Decisions の 3 経路

**Files:**
- Modify: `src/jev_rag/decisions.py`（経路を追加）
- Test: `tests/test_decisions.py`（追加）

**Interfaces:**
- Consumes: Task 3 の `DecisionsClient`、`with_criteria`、`predicate`。`jev_rag.jev.state.candidate_state`、`candidate_path`。`jev_rag.jev.prompts` の `NOUL_INSTRUCTION`、`PLAIN_NOUL_INSTRUCTION`、`RELEVANCE_LEVELS`。`jev_rag.jev.crossencode` の `RELEVANCE`、`GRADE`
- Produces: `DecisionsPointwiseRanker(client, batch_size=10, plain=False, max_workers=8)`（`name` は `decisions_pointwise` か `decisions_pointwise_plain`）、`DecisionsCrossEncodeRanker(client, max_workers=8)`（`name` は `decisions_crossencode`）。どちらも `rank(question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]` と `meter` を持つ

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_decisions.py` の import に `from jev_rag.decisions import DecisionsCrossEncodeRanker, DecisionsPointwiseRanker` と `from jev_rag.types import RetrievedDoc` を足し、末尾に追加:

```python
def candidates(n):
    return [RetrievedDoc(doc_id=f"d{i}", text=f"文書{i}", score=0.0, label=0) for i in range(n)]


def by_position(body):
    """Probability 0.1 * (position + 1), so the order of the result shows the mapping."""
    return [
        {"type": "predicate", "name": q["name"], "probability": 0.1 * (int(q["name"][1:]) + 1)}
        for q in body["questions"]
    ]


def test_pointwise_sends_ten_candidates_per_request_as_json_text():
    sdk = FakeSDK(by_position)
    DecisionsPointwiseRanker(DecisionsClient(sdk)).rank("問い", candidates(12))
    sizes = sorted(len(r["questions"]) for r in sdk.decisions.requests)
    assert sizes == [2, 10]
    first = next(r for r in sdk.decisions.requests if len(r["questions"]) == 10)
    state = json.loads(first["input"])
    assert state["query"] == "問い"
    assert [c["text"] for c in state["candidates"]] == [f"文書{i}" for i in range(10)]


def test_pointwise_questions_point_at_each_candidate_with_the_criteria():
    sdk = FakeSDK(by_position)
    DecisionsPointwiseRanker(DecisionsClient(sdk)).rank("問い", candidates(2))
    questions = sdk.decisions.requests[0]["questions"]
    assert questions[1] == predicate(
        "c1", with_criteria("`candidates[1].text` は `query` に答えるための根拠として役に立ちますか。")
    )


def test_the_plain_route_sends_no_criteria():
    sdk = FakeSDK(by_position)
    ranker = DecisionsPointwiseRanker(DecisionsClient(sdk), plain=True)
    ranker.rank("問い", candidates(1))
    assert ranker.name == "decisions_pointwise_plain"
    assert sdk.decisions.requests[0]["questions"][0]["instructions"] == (
        "`candidates[0].text` は `query` に関連しますか。"
    )


def test_pointwise_sorts_on_the_probability_of_each_document():
    ranked = DecisionsPointwiseRanker(DecisionsClient(FakeSDK(by_position))).rank(
        "問い", candidates(3)
    )
    assert [d.doc_id for d in ranked] == ["d2", "d1", "d0"]
    assert ranked[0].score == pytest.approx(0.3)


def test_a_refused_candidate_scores_zero():
    def reply(body):
        return [{"type": "refusal", "name": "c0"}] + by_position(body)[1:]

    ranked = DecisionsPointwiseRanker(DecisionsClient(FakeSDK(reply))).rank("問い", candidates(2))
    assert {d.doc_id: d.score for d in ranked}["d0"] == 0.0


def test_crossencode_sends_one_request_per_candidate_with_a_predicate_and_a_score():
    def reply(body):
        return [
            {"type": "predicate", "name": "relevance", "probability": 0.6},
            {
                "type": "score",
                "name": "grade",
                "score": 2.0,
                "confidence": 0.5,
                "probabilities": [
                    {"value": i, "label": f"l{i}", "probability": 0.25} for i in range(4)
                ],
            },
        ]

    sdk = FakeSDK(reply)
    ranked = DecisionsCrossEncodeRanker(DecisionsClient(sdk)).rank("問い", candidates(3))
    assert len(sdk.decisions.requests) == 3
    request = sdk.decisions.requests[0]
    assert json.loads(request["input"]).keys() == {"query", "candidate"}
    relevance, grade = request["questions"]
    assert relevance == predicate(
        "relevance", with_criteria("`candidate` は `query` に答えるための根拠として役に立ちますか。")
    )
    assert grade["type"] == "score"
    assert grade["instructions"] == "`candidate` が `query` の根拠としてどの程度有用かを判定してください。"
    assert [level["label"] for level in grade["levels"]] == [
        "無関係。クエリの主題とも参照先とも一致しない。",
        "話題は重なるが、クエリへの回答根拠にはならない。",
        "部分的な根拠を含む。回答の一部を裏づけるが単独では不十分。",
        "直接的な根拠を含む。この記述だけでクエリに答えられる。",
    ]
    assert ranked[0].score == pytest.approx(0.6)
    assert ranked[0].jev_score == pytest.approx(2.0)


def test_every_route_exposes_the_clients_meter():
    client = DecisionsClient(FakeSDK(by_position))
    assert DecisionsPointwiseRanker(client).meter is client.meter
    assert DecisionsCrossEncodeRanker(client).meter is client.meter
```

ファイル先頭に `import pytest` を足す。

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_decisions.py -q`
Expected: FAIL（`ImportError: cannot import name 'DecisionsPointwiseRanker'`）

- [ ] **Step 3: 最小の実装を書く**

`src/jev_rag/decisions.py` の import を次に置き換える:

```python
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from jev_rag.jev.crossencode import GRADE, RELEVANCE
from jev_rag.jev.prompts import (
    NOUL_INSTRUCTION,
    PLAIN_NOUL_INSTRUCTION,
    RELEVANCE_CRITERIA,
    RELEVANCE_LEVELS,
)
from jev_rag.jev.state import candidate_path, candidate_state
from jev_rag.types import RetrievedDoc
from jev_rag.usage import UsageMeter
```

ファイル末尾に追加:

```python
def _probability(answer: dict[str, Any] | None) -> float:
    return float(answer["probability"]) if answer and answer["type"] == "predicate" else 0.0


class DecisionsPointwiseRanker:
    """jev_pointwise on the Decisions API: ten candidates per request, one predicate each.

    Questions are named by position (`c0`, `c1`, ...) rather than by document id,
    so a dataset's id format can never collide with the API's naming rules.
    """

    DEFAULT_BATCH_SIZE = 10

    def __init__(
        self,
        client: DecisionsClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        plain: bool = False,
        max_workers: int = 8,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.plain = plain
        self.max_workers = max_workers
        base = "decisions_pointwise_plain" if plain else "decisions_pointwise"
        self.name = base if batch_size == self.DEFAULT_BATCH_SIZE else f"{base}@{batch_size}"

    @property
    def meter(self) -> UsageMeter:
        return self.client.meter

    def _instruction(self, index: int) -> str:
        if self.plain:
            return PLAIN_NOUL_INSTRUCTION.format(path=candidate_path(index))
        return with_criteria(NOUL_INSTRUCTION.format(path=candidate_path(index)))

    def _judge(self, query: str, group: list[RetrievedDoc]) -> dict[str, float]:
        state, position = candidate_state(query, group)
        questions = [predicate(f"c{position[d.doc_id]}", self._instruction(position[d.doc_id])) for d in group]
        answers = self.client.evaluate(json.dumps(state, ensure_ascii=False), questions)
        return {d.doc_id: _probability(answers.get(f"c{position[d.doc_id]}")) for d in group}

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        groups = [docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)]
        scores: dict[str, float] = {}
        # the batches are independent, as in JevReranker.noul_rerank
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for result in pool.map(lambda group: self._judge(question, group), groups):
                scores.update(result)
        ranked = [replace(doc, score=scores[doc.doc_id]) for doc in docs]
        ranked.sort(key=lambda doc: doc.score, reverse=True)
        return ranked


class DecisionsCrossEncodeRanker:
    """jev_crossencode on the Decisions API: one request per candidate.

    The predicate orders the candidates and the four-level score breaks ties,
    as in jev_rag.jev.crossencode.CrossEncoder.
    """

    name = "decisions_crossencode"

    def __init__(self, client: DecisionsClient, max_workers: int = 8) -> None:
        self.client = client
        self.max_workers = max_workers

    @property
    def meter(self) -> UsageMeter:
        return self.client.meter

    def _judge(self, query: str, doc: RetrievedDoc) -> tuple[float, float | None]:
        questions = [
            predicate("relevance", with_criteria(RELEVANCE)),
            {
                "type": "score",
                "name": "grade",
                "instructions": GRADE,
                "levels": [{"label": level} for level in RELEVANCE_LEVELS],
            },
        ]
        answers = self.client.evaluate(
            json.dumps({"query": query, "candidate": doc.text}, ensure_ascii=False), questions
        )
        grade = answers.get("grade")
        return (
            _probability(answers.get("relevance")),
            float(grade["score"]) if grade and grade["type"] == "score" else None,
        )

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            judgements = list(pool.map(lambda doc: self._judge(question, doc), docs))
        ranked = [
            replace(doc, score=relevance, jev_score=grade)
            for doc, (relevance, grade) in zip(docs, judgements, strict=True)
        ]
        ranked.sort(key=lambda doc: (doc.score, doc.jev_score or 0.0), reverse=True)
        return ranked
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: すべて PASS（整形を求められたら `uv run ruff format src tests`）

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/decisions.py tests/test_decisions.py
git commit -m "feat: add pointwise, plain and cross-encode Decisions routes mirroring the Jev routes"
```

---

### Task 5: 設定、経路の登録、preflight、judge の配線

**Files:**
- Modify: `src/jev_rag/config.py`（`DecisionsSettings`、`Settings.decisions`）
- Modify: `src/jev_rag/rankers.py`（`build_rankers` に `decisions` 引数と 3 経路）
- Modify: `src/jev_rag/preflight.py`（`check_decisions`）
- Modify: `src/jev_rag/cli.py`（`_needs_aws`、`_preflight`、`cmd_judge`）
- Test: `tests/test_rankers.py`、`tests/test_preflight.py`（追加）

**Interfaces:**
- Consumes: Task 3・4 の `DecisionsClient`、`DecisionsPointwiseRanker`、`DecisionsCrossEncodeRanker`
- Produces: `DecisionsSettings().build_client() -> DecisionsClient`、`build_rankers(names, client=None, settings=None, decisions=None)`、`check_decisions(client) -> str`（答えたモデルの ID）、`cli._needs_aws(names) -> bool`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_rankers.py` の末尾に追加:

```python
def test_the_registry_builds_the_decisions_routes_from_their_client():
    from jev_rag.decisions import DecisionsClient

    decisions = DecisionsClient(sdk=None)
    rankers = build_rankers(
        ["decisions_pointwise", "decisions_pointwise_plain", "decisions_crossencode"],
        decisions=decisions,
    )
    assert {name: r.name for name, r in rankers.items()} == {
        "decisions_pointwise": "decisions_pointwise",
        "decisions_pointwise_plain": "decisions_pointwise_plain",
        "decisions_crossencode": "decisions_crossencode",
    }
    assert all(r.meter is decisions.meter for r in rankers.values())
```

`tests/test_preflight.py` の末尾に追加:

```python
def test_a_working_decisions_route_passes_and_reports_the_model():
    from tests.test_decisions import FakeSDK, predicates

    from jev_rag.decisions import DecisionsClient
    from jev_rag.preflight import check_decisions

    assert check_decisions(DecisionsClient(FakeSDK(predicates({"ok": 0.9})))) == "gpt-6-luna"


def test_a_broken_decisions_route_raises_before_the_run_starts():
    from jev_rag.decisions import DecisionsClient
    from jev_rag.preflight import check_decisions

    class Broken:
        class decisions:  # noqa: N801 - mirrors the SDK attribute
            @staticmethod
            def create(**body):
                raise RuntimeError("401 invalid_api_key")

    with pytest.raises(CredentialError, match="OPENAI_API_KEY"):
        check_decisions(DecisionsClient(Broken()))


def test_only_bedrock_routes_need_aws():
    from jev_rag.cli import _needs_aws

    assert _needs_aws(["embedding"]) and _needs_aws(["jev_hybrid"]) and _needs_aws(["cohere_rerank"])
    assert not _needs_aws(["jev_pointwise", "decisions_pointwise", "decisions_crossencode"])
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_rankers.py tests/test_preflight.py -q`
Expected: FAIL（`build_rankers() got an unexpected keyword argument 'decisions'`、`cannot import name 'check_decisions'`、`cannot import name '_needs_aws'`）

- [ ] **Step 3: 最小の実装を書く**

`src/jev_rag/config.py` の `Settings` の直前に追加:

```python
@dataclass
class DecisionsSettings:
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: _env("DECISIONS_MODEL", "gpt-6-luna"))

    def build_client(self):
        from openai import OpenAI

        from jev_rag.decisions import DecisionsClient

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていない。.env に書く")
        # the SDK retries 429 and 5xx with backoff; a full run sends thousands of requests
        return DecisionsClient(OpenAI(api_key=self.api_key, max_retries=5), model=self.model)
```

`Settings` に `decisions: DecisionsSettings = field(default_factory=DecisionsSettings)` を `jev` の次に足す。

`src/jev_rag/rankers.py` の import に `from jev_rag.decisions import DecisionsCrossEncodeRanker, DecisionsPointwiseRanker` を足し、`build_rankers` の定義を `def build_rankers(names: list[str], client: JevClient | None = None, settings=None, decisions=None) -> dict:` に変え、`available` に次の 3 行を `jev_crossencode` の次に足す:

```python
        "decisions_pointwise": lambda: DecisionsPointwiseRanker(decisions),
        "decisions_pointwise_plain": lambda: DecisionsPointwiseRanker(decisions, plain=True),
        "decisions_crossencode": lambda: DecisionsCrossEncodeRanker(decisions),
```

`src/jev_rag/preflight.py` の末尾に追加:

```python
def check_decisions(client: Any) -> str:
    """Confirm the Decisions API answers a trivial question, and return the model that did."""
    import json

    from jev_rag.decisions import predicate

    state = {"query": "赤い惑星", "candidates": [{"text": "火星は赤い惑星と呼ばれる。"}]}
    try:
        client.evaluate(
            json.dumps(state, ensure_ascii=False),
            [predicate("ok", "`candidates[0].text` は `query` の根拠になりますか。")],
        )
    except Exception as error:  # noqa: BLE001 - any failure here is the same failure
        raise CredentialError(
            f"Decisions API へのリクエストが通らない: {error}\n  .env の OPENAI_API_KEY を確認する。"
        ) from error
    return ", ".join(client.meter.snapshot()["models"])
```

`src/jev_rag/cli.py` の `_preflight` を次に置き換え、直前に `_needs_aws` を足す:

```python
def _needs_aws(names: list[str]) -> bool:
    """Routes that call Amazon Bedrock: the embedding, Cohere Rerank, and the hybrid."""
    return any(name in ("embedding", "cohere_rerank", "jev_hybrid") for name in names)


def _preflight(settings, names: list[str], needs_jev: bool) -> None:
    """Fail now rather than forty minutes in.

    A route needing AWS, Jev or the Decisions API is checked before any
    measurement starts, because an expiring credential mid-run costs the whole
    run and reads as a route scoring zero.
    """
    import boto3

    from jev_rag.preflight import check_aws, check_decisions, check_jev

    if _needs_aws(names):
        identity = check_aws(boto3.client("sts", region_name=settings.bedrock.region))
        print(f"AWS: {identity}")
    if needs_jev:
        check_jev(settings.jev.build_client())
        print(f"Jev: {settings.jev.transport} 経路で応答あり")
    if any(name.startswith("decisions") for name in names):
        model = check_decisions(settings.decisions.build_client())
        print(f"Decisions API: {model} が応答")
```

`cmd_judge` の `rankers = build_rankers(...)` の行を次に置き換える:

```python
    decisions = (
        settings.decisions.build_client() if any(n.startswith("decisions") for n in names) else None
    )
    rankers = build_rankers(names, client=client, settings=settings.bedrock, decisions=decisions)
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: すべて PASS

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/config.py src/jev_rag/rankers.py src/jev_rag/preflight.py src/jev_rag/cli.py tests/test_rankers.py tests/test_preflight.py
git commit -m "feat: register the Decisions routes in judge with a credential preflight"
```

---

### Task 6: 順序の入れ替え確認

**Files:**
- Create: `src/jev_rag/probe.py`
- Modify: `src/jev_rag/cli.py`（`probe-order` コマンド）
- Test: `tests/test_probe.py`（新規）

**Interfaces:**
- Consumes: `jev_rag.judge.score_pairs`、`jev_rag.judge_report.mean_query_pr_auc`、`jev_rag.runner.save`、Task 5 の `build_rankers(..., decisions=...)`
- Produces: `reversed_queries(queries) -> list[EvalQuery]`、`order_agreement(forward: Rows, backward: Rows) -> dict`（`pairs`、`median_abs_diff`、`spearman`）、`follows_documents(agreement: dict) -> bool`、`resolves_paths(batched: float, single: float) -> bool`、`run_probe(rankers: dict, queries, out) -> dict`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_probe.py`:

```python
"""Do a batched route's probabilities follow the documents or their positions?"""

import pytest

from jev_rag.dataset import EvalQuery
from jev_rag.probe import follows_documents, order_agreement, resolves_paths, reversed_queries, run_probe
from jev_rag.types import RetrievedDoc


def queries():
    return [
        EvalQuery(
            query_id=f"q{i}",
            question="問",
            candidates=[
                RetrievedDoc(doc_id=f"q{i}-d{j}", text=f"文書{j}", score=0.0, label=int(j == 0))
                for j in range(4)
            ],
        )
        for i in range(3)
    ]


class ByDocument:
    """A judge that reads the document: the relevant one scores high wherever it sits."""

    name = "by_document"

    def rank(self, question, docs):
        for doc in docs:
            doc.score = 0.9 if doc.label else 0.1 + 0.01 * int(doc.doc_id[-1])
        return docs


class ByPosition:
    """A judge that only sees positions: the first slot always scores high."""

    name = "by_position"

    def rank(self, question, docs):
        for position, doc in enumerate(docs):
            doc.score = 0.9 - 0.2 * position
        return docs


def test_reversing_keeps_every_document_and_flips_the_order():
    flipped = reversed_queries(queries())
    assert [d.doc_id for d in flipped[0].candidates] == ["q0-d3", "q0-d2", "q0-d1", "q0-d0"]
    assert [d.doc_id for d in queries()[0].candidates] == ["q0-d0", "q0-d1", "q0-d2", "q0-d3"]


def test_a_judge_that_reads_documents_agrees_with_itself_in_both_orders(tmp_path):
    result = run_probe({"by_document": ByDocument()}, queries(), tmp_path / "p.json")["by_document"]
    assert result["median_abs_diff"] == pytest.approx(0.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["follows_documents"] is True


def test_a_judge_that_reads_positions_fails_the_check(tmp_path):
    result = run_probe({"by_position": ByPosition()}, queries(), tmp_path / "p.json")["by_position"]
    assert result["follows_documents"] is False
    assert (tmp_path / "p.json").exists()


def test_agreement_is_computed_per_document_not_per_position():
    forward = [{"query_id": "q", "doc_id": "a", "score": 0.9}, {"query_id": "q", "doc_id": "b", "score": 0.1}]
    backward = [{"query_id": "q", "doc_id": "b", "score": 0.1}, {"query_id": "q", "doc_id": "a", "score": 0.9}]
    assert order_agreement(forward, backward)["median_abs_diff"] == pytest.approx(0.0)


def test_the_thresholds_are_the_ones_in_the_spec():
    assert follows_documents({"median_abs_diff": 0.1, "spearman": 0.8}) is True
    assert follows_documents({"median_abs_diff": 0.11, "spearman": 0.9}) is False
    assert follows_documents({"median_abs_diff": 0.05, "spearman": 0.79}) is False
    assert resolves_paths(batched=0.70, single=0.90) is True
    assert resolves_paths(batched=0.69, single=0.90) is False
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_probe.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'jev_rag.probe'`）

- [ ] **Step 3: 最小の実装を書く**

`src/jev_rag/probe.py`:

```python
"""Do a route's probabilities follow the documents or their positions?

A batched route names each candidate by its position in the request
(`candidates[3].text`). If the model does not resolve those paths, reversing
the candidates moves the probabilities with the positions, not with the
documents. strands-decider failed exactly this way. Each route is run on the
same questions in both orders and the per-document probabilities are compared.

Thresholds are fixed in .claude/docs/specs/openai-decisions-comparison.md.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from jev_rag.dataset import EvalQuery
from jev_rag.judge import score_pairs
from jev_rag.judge_report import mean_query_pr_auc
from jev_rag.runner import save

MAX_MEDIAN_DIFF = 0.1
MIN_SPEARMAN = 0.8
MAX_BATCH_GAP = 0.2


def reversed_queries(queries: list[EvalQuery]) -> list[EvalQuery]:
    return [replace(q, candidates=list(reversed(q.candidates))) for q in queries]


def order_agreement(forward: list[dict[str, Any]], backward: list[dict[str, Any]]) -> dict[str, Any]:
    other = {(r["query_id"], r["doc_id"]): r["score"] for r in backward}
    pairs = [(r["score"], other[(r["query_id"], r["doc_id"])]) for r in forward]
    a, b = (np.array(side, dtype=float) for side in zip(*pairs, strict=True))
    rho = spearmanr(a, b).statistic
    return {
        "pairs": len(pairs),
        "median_abs_diff": float(np.median(np.abs(a - b))),
        # a constant score has no rank order; nan fails the check below
        "spearman": float(rho),
    }


def follows_documents(agreement: dict[str, Any]) -> bool:
    rho = agreement["spearman"]
    return (
        agreement["median_abs_diff"] <= MAX_MEDIAN_DIFF
        and not math.isnan(rho)
        and rho >= MIN_SPEARMAN
    )


def resolves_paths(batched: float, single: float) -> bool:
    """A batched route far below its one-at-a-time counterpart is not reading the paths."""
    return round(single - batched, 10) <= MAX_BATCH_GAP


def run_probe(rankers: dict[str, Any], queries: list[EvalQuery], out: str | Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    flipped = reversed_queries(queries)
    for name, ranker in rankers.items():
        forward, backward = score_pairs(ranker, queries), score_pairs(ranker, flipped)
        agreement = order_agreement(forward["pairs"], backward["pairs"])
        results[name] = agreement | {
            "follows_documents": follows_documents(agreement),
            "mean_query_pr_auc": mean_query_pr_auc(forward["pairs"]),
            "failed": forward["failed"] + backward["failed"],
            "forward": forward["pairs"],
            "backward": backward["pairs"],
        }
        save(results, Path(out))
    return results
```

`src/jev_rag/cli.py` に `cmd_judge` の次へ追加:

```python
def cmd_probe_order(args: argparse.Namespace) -> int:
    """Send the same questions in two candidate orders and compare per-document probabilities."""
    from jev_rag.config import Settings
    from jev_rag.probe import resolves_paths, run_probe

    settings = Settings()
    queries = sample_queries(_judge_queries(args.dataset), args.queries, seed=args.seed)
    names = [n for n in args.rankers.split(",") if n]
    needs_jev = any(n.startswith("jev") for n in names)
    _preflight(settings, names, needs_jev=needs_jev)
    client = settings.jev.build_client() if needs_jev else None
    decisions = (
        settings.decisions.build_client() if any(n.startswith("decisions") for n in names) else None
    )
    rankers = build_rankers(names, client=client, settings=settings.bedrock, decisions=decisions)
    results = run_probe(rankers, queries, args.out)
    print(f"{args.out} に保存しました\n")
    print("| 経路 | 組 | 差の中央値 | スピアマン | 文書に付いてくる | 問題ごとの PR-AUC | 失敗 |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(
            f"| {name} | {r['pairs']} | {r['median_abs_diff']:.3f} | {r['spearman']:.3f} | "
            f"{'はい' if r['follows_documents'] else 'いいえ'} | {r['mean_query_pr_auc']:.3f} | "
            f"{len(r['failed'])} |"
        )
    for batched, single in (
        ("decisions_pointwise", "decisions_crossencode"),
        ("jev_pointwise", "jev_crossencode"),
    ):
        if batched in results and single in results:
            ok = resolves_paths(
                results[batched]["mean_query_pr_auc"], results[single]["mean_query_pr_auc"]
            )
            print(f"\n{batched} は {single} との差が 0.2 以内か: {'はい' if ok else 'いいえ'}")
    return 0
```

`main` の `judge.set_defaults(func=cmd_judge)` の次に追加:

```python
    probe = sub.add_parser(
        "probe-order", help="候補の順序を逆にして送り、確率が文書に付いてくるかを確かめる"
    )
    probe.add_argument("--dataset", choices=["synthetic", "jragbench", "miracl"], default="synthetic")
    probe.add_argument(
        "--rankers",
        default="decisions_pointwise,decisions_crossencode,jev_pointwise,jev_crossencode",
    )
    probe.add_argument("--queries", type=int, default=5)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--out", default="out/decisions/probe-order.json")
    probe.set_defaults(func=cmd_probe_order)
```

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: すべて PASS（`jev_rag.runner.save` は親ディレクトリを作るので、`out/decisions/` を先に作る必要はない）

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/probe.py src/jev_rag/cli.py tests/test_probe.py
git commit -m "feat: add probe-order to check whether batched probabilities follow the documents"
```

---

### Task 7: report-judge の「Decisions と Jev 1.14」の節

**Files:**
- Modify: `src/jev_rag/judge_report.py`
- Test: `tests/test_judge_report.py`（追加）

**Interfaces:**
- Consumes: Task 2 の `scored["usage"]`、既存の `_pairs`、`_query_pr_aucs`、`_interval_of_means`、`paired_bootstrap`、`pooled_pr_auc`、`judge_metrics`
- Produces: `_decisions_section(judges, samples, seed) -> list[str]`（`format_report` から呼ぶ）。定数 `PRICE_PER_MTOK`、`DECISIONS_PAIRS`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_judge_report.py` の末尾に追加:

```python
def measured(rows_, models, tokens, refusals=0):
    return {
        "scored": {
            "pairs": rows_,
            "failed": [],
            "usage": {
                "requests": 1,
                "input_tokens": tokens,
                "refusals": refusals,
                "missing": 0,
                "models": models,
                "request_seconds": [0.1],
            },
        }
    }


def decisions_judges():
    jev = rows("q1", [(0.9, 1), (0.1, 0)]) + rows("q2", [(0.8, 1), (0.3, 0)])
    decisions = rows("q1", [(0.7, 1), (0.6, 0)]) + rows("q2", [(0.4, 1), (0.5, 0)])
    return {
        "synthetic": {
            "jev_pointwise": measured(jev, ["jev-1.14.0"], 1_000_000),
            "decisions_pointwise": measured(decisions, ["gpt-6-luna"], 2_000_000, refusals=1),
        }
    }


def test_the_report_compares_decisions_with_jev():
    report = format_report(decisions_judges(), None, samples=50)
    assert "### Decisions と Jev 1.14" in report
    assert "jev-1.14.0" in report and "gpt-6-luna" in report
    # Jev bills $0.042 and Decisions $0.10 per million input tokens
    assert "0.0420" in report and "0.2000" in report
    assert "| 1/0/0 |" in report
    assert "Decisions − Jev" in report


def test_the_decisions_section_is_left_out_without_decisions_routes():
    judges = decisions_judges()
    del judges["synthetic"]["decisions_pointwise"]
    assert "Decisions と Jev" not in format_report(judges, None, samples=50)
```

- [ ] **Step 2: テストが失敗することを確かめる**

Run: `uv run pytest tests/test_judge_report.py -q`
Expected: FAIL（`### Decisions と Jev 1.14` が出力に無い）

- [ ] **Step 3: 最小の実装を書く**

`src/jev_rag/judge_report.py` の import を `from jev_rag.judge import judge_metrics, paired_bootstrap, pooled_pr_auc` に変え、`_dilution_section` の直前に追加:

```python
# USD per million input tokens, from the official pages fetched on 2026-10-19
PRICE_PER_MTOK = {"jev": 0.042, "decisions": 0.10}
DECISIONS_PAIRS = [
    ("jev_pointwise", "decisions_pointwise", "10 件ずつ、判定基準あり"),
    ("jev_pointwise_plain", "decisions_pointwise_plain", "10 件ずつ、判定基準なし"),
    ("jev_crossencode", "decisions_crossencode", "1 件ずつ"),
]


def _decisions_section(judges, samples, seed) -> list[str]:
    body, notes = [], []
    for name, result in judges.items():
        label = DATASETS.get(name, name)
        for jev_route, decisions_route, shape in DECISIONS_PAIRS:
            jev, decisions = _pairs(result, jev_route), _pairs(result, decisions_route)
            if not jev or not decisions:
                continue
            for route, rows in ((jev_route, jev), (decisions_route, decisions)):
                scored = result[route]["scored"]
                usage = scored.get("usage") or {}
                tokens = usage.get("input_tokens")
                price = PRICE_PER_MTOK["jev" if route.startswith("jev") else "decisions"]
                body.append(
                    [
                        label,
                        shape,
                        f"`{route}`",
                        ", ".join(usage.get("models", [])) or "-",
                        _num(mean_query_pr_auc(rows), 3),
                        _num(pooled_pr_auc(rows), 2),
                        _num(judge_metrics(rows, probabilistic=True).get("ece"), 3),
                        f"{usage.get('refusals', 0)}/{usage.get('missing', 0)}/{len(scored['failed'])}",
                        "-" if tokens is None else f"{tokens:,}",
                        "-" if tokens is None else f"{tokens / 1e6 * price:.4f}",
                    ]
                )
            a, b = _query_pr_aucs(decisions), _query_pr_aucs(jev)
            low, high = _interval_of_means(a, b, samples, seed)
            ci = paired_bootstrap(decisions, jev, samples=samples, seed=seed)
            notes.append(
                f"{label}、{shape}: Decisions − Jev は問題ごとの PR-AUC "
                f"{_diff(mean_query_pr_auc(decisions), mean_query_pr_auc(jev), 3)}"
                f"（{_interval(low, high, 3)}）、全問まとめ PR-AUC "
                f"{_diff(pooled_pr_auc(decisions), pooled_pr_auc(jev), 2)}"
                f"（{_interval(ci['low'], ci['high'], 2)}）"
            )
    if not body:
        return []
    header = [
        "データ",
        "形",
        "経路",
        "答えたモデル",
        "問題ごとの PR-AUC",
        "全問まとめ PR-AUC",
        "ECE",
        "拒否/欠け/失敗した問題",
        "入力トークン",
        "費用（ドル）",
    ]
    return ["### Decisions と Jev 1.14", ""] + _table(header, body) + [""] + notes
```

`format_report` の `sections` のリストで `_criteria_section(...)` の次に `_decisions_section(judges, samples, seed),` を足す。

- [ ] **Step 4: テストが通ることを確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: すべて PASS。`uv run jev-rag report-judge --results-dir results` の出力が変更前と同じ（`results/` に Decisions の経路が無いので新しい節は出ない）ことも確かめる。変更前の出力は Task 7 の着手前に `uv run jev-rag report-judge --results-dir results > out/report-before.md` で保存しておき（`out/` は追跡対象外）、変更後の出力と `command diff` で比べる（zsh の `diff` 関数は壊れているので `command` を付ける）

- [ ] **Step 5: Commit**

```bash
git add src/jev_rag/judge_report.py tests/test_judge_report.py
git commit -m "feat: report Decisions against Jev 1.14 with intervals, refusals and cost"
```

---

### Task 8: 順序の入れ替え確認を実行する（API を呼ぶ）

**Files:**
- Modify: `.env`（`JEV_DIRECT_MODEL=jev-1.14.0`。追跡対象外）
- Create: `.claude/docs/research/decisions-probe.md`

**Interfaces:**
- Consumes: Task 6 の `jev-rag probe-order`

- [ ] **Step 1: Jev のモデルを 1.14 に固定する**

`.env` の `JEV_DIRECT_MODEL=` の行を `JEV_DIRECT_MODEL=jev-1.14.0` に書き換える。`JEV_TRANSPORT=direct` であることも確かめる。

Run: `grep -E '^(JEV_TRANSPORT|JEV_DIRECT_MODEL)=' .env`
Expected: `JEV_TRANSPORT=direct` と `JEV_DIRECT_MODEL=jev-1.14.0`

- [ ] **Step 2: 合成データが手元にあることを確かめる**

Run: `ls data/synthetic/gen_a | head -3`
Expected: `a-001.json` などが表示される

- [ ] **Step 3: 確認を実行する**

Run: `uv run jev-rag probe-order`
Expected: preflight で `Jev: direct 経路で応答あり` と `Decisions API: gpt-6-luna が応答` が出て、4 経路の表と 2 行の判定が出る。`out/decisions/probe-order.json` が作られる

- [ ] **Step 4: 答えたモデルを確かめる**

Run: `uv run python -c "import json; d=json.load(open('out/decisions/probe-order.json')); print({k: len(v['failed']) for k, v in d.items()})"`
Expected: どの経路も失敗 0。Jev 1.14 で答えたかは、Task 9 の `usage.models` で `jev-1.14.0` を確かめる（probe の結果には usage を保存しない）

- [ ] **Step 5: 結果を記録する**

`.claude/docs/research/decisions-probe.md` に、実行日、実行したコマンド、4 経路の表（Step 3 の出力をそのまま）、2 行の判定、そこから決めた本測定の経路を書く。判定のルールは次のとおり。

- `decisions_pointwise` が「文書に付いてくる: はい」かつ「差が 0.2 以内: はい」なら、Decisions の 3 経路をすべて本測定する
- どちらかが「いいえ」なら、`decisions_pointwise` と `decisions_pointwise_plain` を外し、`decisions_crossencode` だけを本測定する。外した理由と数値を書く
- Jev の 2 経路の結果も同じ表に残す（Jev 1.14 でもまとめる経路が成り立つかの確認）

- [ ] **Step 6: Commit**

```bash
git add .claude/docs/research/decisions-probe.md
git commit -m "docs: record the candidate-order probe for the Decisions and Jev routes"
```

---

### Task 9: 本測定と集計（API を呼ぶ）

**Files:**
- Create: `.claude/docs/research/decisions-results.md`

**Interfaces:**
- Consumes: Task 5 の `jev-rag judge`、Task 7 の `report-judge`、Task 8 で決めた経路

- [ ] **Step 1: 測る経路を決める**

Task 8 の結果に従い、`ROUTES` を次のどちらかにする。

- 確認を通った場合: `jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_pointwise,decisions_pointwise_plain,decisions_crossencode`
- 通らなかった場合: `jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_crossencode`

- [ ] **Step 2: 小さいデータから順に測る**

Run（`<ROUTES>` は Step 1 の文字列）:

```bash
uv run jev-rag judge --dataset synthetic --rankers <ROUTES> --out out/decisions/judge-synthetic.json
uv run jev-rag judge --dataset jragbench --rankers <ROUTES> --out out/decisions/judge-jragbench.json
uv run jev-rag judge --dataset miracl --rankers <ROUTES> --out out/decisions/judge-miracl.json
```

Expected: 各コマンドの最後に経路ごとの表が出て、「拒否/欠け」の列が出る。MIRACL は時間がかかるので `run_in_background` で実行する。途中で止まっても、同じコマンドを再実行すれば測定済みの経路は飛ばして続きから測る

- [ ] **Step 3: 失敗、拒否、答えたモデルを確かめる**

Run:

```bash
uv run python - <<'EOF'
import json
for ds in ("synthetic", "jragbench", "miracl"):
    d = json.load(open(f"out/decisions/judge-{ds}.json"))
    for name, r in d.items():
        s = r.get("scored", {})
        u = s.get("usage")
        if u:
            print(ds, name, "failed", len(s["failed"]), "refusals", u["refusals"], "missing", u["missing"], u["models"], u["input_tokens"])
EOF
```

Expected: Jev の経路の `models` が `['jev-1.14.0']`、Decisions の経路が `gpt-6-luna` で始まる ID。失敗した問題があれば、同じ `judge` コマンドをもう一度実行する前に、その経路のエントリを JSON から消すかどうかをユーザーに確認する（`run_judge` は `pr_auc` がある経路を飛ばすので、一部失敗の経路は再実行されない）

- [ ] **Step 4: 集計する**

Run: `uv run jev-rag report-judge --results-dir out/decisions`
Expected: 「### Decisions と Jev 1.14」の節だけが出る（`cohere_rerank` が無いので既存の節は出ない）

- [ ] **Step 5: 結果を記録する**

`.claude/docs/research/decisions-results.md` に、測定日、Jev と Decisions の答えたモデルの ID、実行したコマンド、Step 4 の出力（表と差の行）、Step 3 の失敗・拒否・欠けの件数、各経路の所要時間（`scored.seconds`。結論にはしない旨を添える）を書く。読み方は、測った値から言えることだけを書く。

- [ ] **Step 6: Commit**

```bash
git add .claude/docs/research/decisions-results.md
git commit -m "docs: record the Decisions API versus Jev 1.14 judge results"
```

---

### Task 10: README と引き継ぎノート

**Files:**
- Modify: `README.md`
- Modify: `.claude/handovers/`（handover スキルで作成・更新）

- [ ] **Step 1: README に節を足す**

`README.md` の「## Jev の 2 経路」の直前に次の節を足す。経路の行は Task 9 で実際に測ったものだけを書く。

```markdown
## Decisions API との比較

OpenAI の Decisions API（`gpt-6-luna`、public beta）を、Jev 1.14（`jev-1.14.0`）と同じデータ、指示文、指標で比べます。設計は `.claude/docs/specs/openai-decisions-comparison.md`、結果は `.claude/docs/research/decisions-results.md` にあります。

| 経路 | 中身 |
|---|---|
| `decisions_pointwise` | 候補文書 10 件を JSON の文字列にして 1 リクエストにまとめ、判定基準を指示文の末尾に足した `predicate` |
| `decisions_pointwise_plain` | 判定基準なしの `predicate` |
| `decisions_crossencode` | 候補文書 1 件ごとに 1 リクエスト。`predicate`（関連）と `score`（4 段階）の組 |

`.env` に `OPENAI_API_KEY` を書き、`JEV_DIRECT_MODEL=jev-1.14.0` にしてから実行します。

```bash
uv run jev-rag probe-order   # 候補の順序を逆にして、確率が文書に付いてくるかを確かめる
uv run jev-rag judge --dataset synthetic --rankers jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_pointwise,decisions_pointwise_plain,decisions_crossencode --out out/decisions/judge-synthetic.json
uv run jev-rag report-judge --results-dir out/decisions
```

Decisions には Jev の `criteria` にあたる欄がないため、判定基準は指示文の文面に入れています。この点は Jev と同じ条件ではありません。
```

（README の中にコードブロックを入れ子で書くので、実際の README では外側の ```markdown の囲みは付けない。）

- [ ] **Step 2: 確かめる**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests && git status --short`
Expected: すべて PASS。`git status` に出るのは `README.md` と untracked の `CLAUDE.md` だけ

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe the Decisions API comparison in the README"
```

- [ ] **Step 4: 引き継ぎノートを更新する**

handover スキルで、このセッションの引き継ぎノートを作成（または同じセッションのノートを更新）し、測定の結果と残りの作業（push、PR、Issue、記事）を書いて commit する。push と PR の作成はユーザーの承認を得てから行う。
