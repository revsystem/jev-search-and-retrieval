# OpenAI Decisions API と Jev 1.14 の判定器比較の仕様

2026-10-19 にユーザーと合意した設計。ブランチは `feat/openai-decisions-comparison`。

## 目的

OpenAI が public beta で出した Decisions API（`gpt-6-luna`）を、RAG の rerank の判定器として使ったときに、TypeSafe Jev 1.14 と比べて関連文書をどれだけ正しく見分けられるかを測る。データ、指示文と判定基準、指標の計算は、記事（Jev と Cohere Rerank の比較）で使ったものをそのまま使う。

## 両 API の違い

Decisions API の仕様は `.claude/docs/research/openai-decisions/`、Jev 1.14 は `.claude/docs/research/typesafe/models.md` にある（どちらも 2026-10-19 取得）。

| | Jev 1.14 | Decisions API |
|---|---|---|
| エンドポイント | `POST /v1/systemone` | `POST /v1/decisions` |
| モデル | `jev-1.14.0`（`jev-latest` はまだ `jev-1.13.0` を指す） | `gpt-6-luna` のみ。バージョン付きの ID はない |
| 入力 | `state`（文字列、JSON オブジェクト、テキストの配列） | `input`（文字列、またはテキストと画像を含む user メッセージ） |
| 確率を返す質問 | Noul（`criteria` で true/false の定義を渡せる） | `predicate`（`instructions` だけ） |
| 段階の質問 | Score（`criteria` に段階の文のリスト） | `score`（`levels` に `label` と任意の `description`） |
| 1 リクエストの上限 | state と全質問で 64k、state と 1 問で 32k トークン | 質問 1〜200 問。入力の上限は Decisions のページに記述なし（モデルの最大入力は 922,000 トークン） |
| 答えない場合 | 記述なし | 質問ごとに `refusal` が返りうる |
| 料金 | 入力 $0.042/1M トークン | 入力 $0.10/1M トークン。出力とキャッシュは課金なし |
| 応答の欄 | `answers`、`model`（バージョン付き ID）、`usage`。所要時間の欄はない（保存済みの `api.md` による） | `answers`、`model`、`usage`。所要時間の欄はない |

同じ条件にできない点が 2 つあり、記事にも明記する。

1. Decisions には `criteria` の欄がない。判定基準付きの経路では、`RELEVANCE_CRITERIA` の true/false の定義を `instructions` の末尾に文として足す。
2. Decisions のドキュメントには、構造化した入力も `candidates[3].text` のようなパスでの参照も書かれていない。JSON を文字列にして渡し、指示文のパスをモデルが解釈できるかは、本測定の前の確認（後述）で確かめる。

## 経路

| 経路 | 中身 |
|---|---|
| `jev_pointwise` | 既存。1 問の候補 10 件を 1 つの state にまとめ、判定基準付き Noul |
| `jev_pointwise_plain` | 既存。判定基準なし |
| `jev_crossencode` | 既存。1 件 1 リクエスト、Noul（関連）と Score（4 段階）の組 |
| `decisions_pointwise` | 候補 10 件を 1 リクエストにまとめ、判定基準を文面に足した `predicate` |
| `decisions_pointwise_plain` | 判定基準なしの `predicate` |
| `decisions_crossencode` | 1 件 1 リクエスト、`predicate`（関連）と `score`（4 段階）の組 |

Jev の経路は `JEV_DIRECT_MODEL=jev-1.14.0` で測り直す。埋め込みと Cohere Rerank は測り直さない。

### Decisions のリクエストの形

指示文は `jev/prompts.py` と `jev/crossencode.py` の文字列を変えずに使う。

まとめる経路（`decisions_pointwise`）の例:

```json
{
  "model": "gpt-6-luna",
  "input": "{\"query\": \"…\", \"candidates\": [{\"text\": \"…\"}, …]}",
  "questions": [
    {
      "type": "predicate",
      "name": "<chunk_id>",
      "instructions": "`candidates[0].text` は `query` に答えるための根拠として役に立ちますか。\n判定基準: はい（true）= クエリが指す対象そのものを扱い、回答を裏づける事実・数値・定義を含む。いいえ（false）= 話題が重なるだけ、参照先が異なる、または根拠となる情報を含まない。"
    }
  ]
}
```

- `input` は `jev_rag.jev.state.candidate_state` が作る state を `json.dumps(ensure_ascii=False)` した文字列
- `_plain` の経路は `PLAIN_NOUL_INSTRUCTION` だけを送り、判定基準の文を足さない
- 1 リクエストの件数は Jev と同じく 10 件（`JevPointwiseRanker.DEFAULT_BATCH_SIZE`）
- 1 件ずつの経路の `input` は `{"query": …, "candidate": …}`。質問は `crossencode.RELEVANCE`（判定基準を足した `predicate`）と `crossencode.GRADE`（`RELEVANCE_LEVELS` の 4 文を順に `levels[].label` にした `score`）
- 並べ替えには `predicate` の確率を使い、同点は `score` で決める（Jev の `CrossEncoder` と同じ）

### 応答の扱い

- 答えは `name` で質問に対応づける。順序には頼らない
- `refusal` と、応答に含まれない質問は、スコアを 0.0 にする（Jev の既存の扱いと同じ）。そのうえで件数を経路とデータごとに保存し、`judge` と `report-judge` に表示する。Jev 側の欠けた答えも同じく数える
- リクエストが失敗した問題は、既存の `score_pairs` と同じく `failed` に記録する
- 経路ごとに、応答の `model` 欄（答えたモデルの ID）と `usage.input_tokens` の合計を保存する。Jev 側にも同じ記録を足す
- 並列度は Jev と同じ 8。リクエストごとの所要時間を記録するが、速さの優劣は結論にしない

## 本測定の前の確認

strands-decider の比較では、10 件まとめた経路がパスを解釈できず、当て推量の水準まで落ちた。Decisions でも同じことが起きうるので、本測定の前に次を確かめる。

- 合成データから 5 問（50 組）を選び、`decisions_pointwise` と `jev_pointwise` で、候補をそのままの順と逆順の 2 通りで送る
- 文書ごとに 2 回の確率を対応づけ、次の 2 つをどちらも満たせば「確率が文書に付いてくる」とみなす
  - 2 回の確率の差の絶対値の中央値が 0.1 以下
  - 2 回の確率のスピアマン順位相関が 0.8 以上
- 同じ 5 問で `decisions_crossencode` も測る。まとめる経路の問題ごと PR-AUC の平均が、1 件ずつの経路より 0.2 以上低ければ、パスを解釈できていないとみなす
- 満たさない場合は、`decisions_pointwise` と `decisions_pointwise_plain` を本測定から外し、`decisions_crossencode` だけを測る。外した理由と確認の数値は記事に書く

確認の結果は `.claude/docs/research/decisions-probe.md` に残す。

## 集計

`report-judge` に「Decisions と Jev 1.14」の節を足す。データごとに次を出す。

- 問題ごとの PR-AUC の平均（MIRACL は候補がすべて関連文書の問題を除く。既存の表と同じ）
- 全組をまとめた PR-AUC
- Decisions − Jev の差と 95% 区間（問題を単位に 2,000 回の bootstrap、`--seed 0`。既存の `paired_bootstrap`）
- ECE（両経路とも確率を返すので計算する）
- `refusal` と欠けた答えの件数、失敗した問題の数
- `usage.input_tokens` の合計と、そこから計算した費用（料金は 2026-10-19 時点の公式の値）

## 出力先

`judge --out out/decisions/judge-<dataset>.json` に保存する。記事の測定結果（`results/`）と既存の `out/judge-*.json` は変えない。結果が固まったら、記事用に `results/decisions/` へ写すかを決める。

## 設定と依存

- `pyproject.toml` に `openai>=3.26.0` を足す（Decisions に対応した最初のバージョン。ガイドの記述）
- `.env` に `OPENAI_API_KEY`（追加済み）と `JEV_DIRECT_MODEL=jev-1.14.0`
- `preflight` に Decisions の疎通確認を足し、Decisions の経路を含む実行の前に呼ぶ
- untracked の `uv.lock` が書き換わる。commit に含めるかはユーザーに確認する

## テスト

TDD で書く。API を呼ばない偽のクライアントを使う。

- リクエストの中身（`input` の JSON、判定基準を足した `instructions`、`name`、`_plain` で判定基準が付かないこと、`levels` の順序）
- `name` での対応づけ（応答の順序が入れ替わっても正しい文書に付くこと）
- `refusal` と欠けた答えが 0.0 になり、件数に数えられること
- 経路が `build_rankers` に登録され、Decisions の経路だけを含む実行で AWS の preflight を求めないこと
- `model` 欄と `usage.input_tokens` が保存されること
- `report-judge` の新しい節が、保存済みの結果から計算できること

## やらないこと

- 件数を変える測定（`dilution`）は、まとめる経路が確認を通った場合だけ、別に検討する
- 画像の入力、`choice` 型の質問、プロンプトキャッシュの設定は使わない
