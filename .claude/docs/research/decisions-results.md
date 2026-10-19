# Decisions API と Jev 1.14 の判定器比較の結果

2026-10-19 に測定した。設計は `.claude/docs/specs/openai-decisions-comparison.md`、本測定の前の確認は `decisions-probe.md` にある。

## 条件

| | Jev | Decisions API |
|---|---|---|
| 答えたモデル（応答の `model` 欄） | `jev-1.14.0` | `gpt-6-luna` |
| 測った経路 | `jev_pointwise`、`jev_pointwise_plain`、`jev_crossencode` | `decisions_crossencode` |
| 並列度 | 8 | 8 |
| 試行回数とタイムアウト | 3 回、60 秒 | 3 回（SDK の `max_retries=2`）、60 秒 |

Decisions の 10 件まとめる 2 経路は、候補の順序を逆にすると確率が大きく動いたため、本測定から外した（`decisions-probe.md`）。データ、指示文、判定基準の文面、指標の計算は、記事（Jev と Cohere Rerank の比較）と同じ。Decisions には `criteria` の欄がないので、判定基準は指示文の末尾に入れた。

```bash
ROUTES=jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_crossencode
uv run jev-rag judge --dataset synthetic --rankers $ROUTES --out out/decisions/judge-synthetic.json
uv run jev-rag judge --dataset jragbench --rankers $ROUTES --out out/decisions/judge-jragbench.json
uv run jev-rag judge --dataset miracl --rankers $ROUTES --out out/decisions/judge-miracl.json
uv run jev-rag report-judge --results-dir out/decisions
```

最初の MIRACL の測定では、`jev_crossencode` の 54 問が失敗した。その後の再実行が `402 billing_error`（TypeSafe のクレジット切れ）で止まったので、失敗もクレジット切れによると考えられる（失敗した時点の応答本文は残っていないので未確認）。ユーザーがクレジットを追加してから `jev_crossencode` だけを測り直し、失敗は 0 問になった。最初の結果は `out/decisions/judge-miracl.before-rerun.json` に残してある。

すべての経路で、拒否、欠けた答え、失敗した問題は 0 件だった。Jev のまとめる経路のリクエスト数は、3 つのデータとも「問題ごとの候補数を 10 件ずつに分けた数」と一致した（MIRACL 1,134、J-RAGBench 117、合成 60）。入力の上限による分割は起きていない。

## 1 件ずつの経路どうしの比較

`report-judge --results-dir out/decisions` の出力:

| データ | 経路 | 答えたモデル | 問題ごとの PR-AUC | 全問まとめ PR-AUC | ECE | 拒否/欠け/失敗した問題 | 入力トークン | 費用（ドル） |
|---|---|---|---|---|---|---|---|---|
| MIRACL | `jev_crossencode` | jev-1.14.0 | 0.862 | 0.75 | 0.160 | 0/0/0 | 7,325,018 | 0.3077 |
| MIRACL | `decisions_crossencode` | gpt-6-luna | 0.799 | 0.68 | 0.133 | 0/0/0 | 5,976,555 | 0.5977 |
| J-RAGBench | `jev_crossencode` | jev-1.14.0 | 0.905 | 0.79 | 0.069 | 0/0/0 | 855,195 | 0.0359 |
| J-RAGBench | `decisions_crossencode` | gpt-6-luna | 0.777 | 0.71 | 0.112 | 0/0/0 | 701,465 | 0.0701 |
| 合成データ | `jev_crossencode` | jev-1.14.0 | 0.840 | 0.74 | 0.078 | 0/0/0 | 480,386 | 0.0202 |
| 合成データ | `decisions_crossencode` | gpt-6-luna | 0.826 | 0.73 | 0.157 | 0/0/0 | 397,627 | 0.0398 |

Decisions − Jev（問題を単位に 2,000 回の bootstrap、95% 区間）:

| データ | 比べた問題 | 問題ごとの PR-AUC | 全問まとめ PR-AUC |
|---|---|---|---|
| MIRACL | 797 問 | −0.063（−0.075〜−0.049） | −0.07（−0.08〜−0.05） |
| J-RAGBench | 60 問 | −0.128（−0.184〜−0.075） | −0.08（−0.14〜−0.02） |
| 合成データ | 60 問 | −0.014（−0.048〜+0.020） | −0.01（−0.05〜+0.02） |

MIRACL は、候補がすべて関連文書の 63 問を除いた 797 問で比べた（既存の表と同じ扱い）。

## Jev のまとめる経路を含めた比較

Decisions で測れたのは 1 件ずつの経路だけだが、Jev は 10 件まとめる経路の方が成績がよい。両 API で最も成績のよい経路どうしの差も、保存したスコアから同じ方法で計算した（`judge_report` の関数を直接呼んだ。report-judge の表にはない）。

| データ | 経路 | 問題ごとの PR-AUC | 全問まとめ PR-AUC | 関連文書の 9 割を残したときの関連の割合 |
|---|---|---|---|---|
| MIRACL | `jev_pointwise` | 0.866 | 0.761 | 59.4% |
| MIRACL | `jev_pointwise_plain` | 0.849 | 0.729 | 54.2% |
| MIRACL | `jev_crossencode` | 0.862 | 0.749 | 57.6% |
| MIRACL | `decisions_crossencode` | 0.799 | 0.680 | 56.0% |
| J-RAGBench | `jev_pointwise` | 0.938 | 0.871 | 62.0% |
| J-RAGBench | `jev_pointwise_plain` | 0.948 | 0.872 | 56.6% |
| J-RAGBench | `jev_crossencode` | 0.905 | 0.787 | 53.9% |
| J-RAGBench | `decisions_crossencode` | 0.777 | 0.711 | 20.7% |
| 合成データ | `jev_pointwise` | 0.952 | 0.921 | 84.1% |
| 合成データ | `jev_pointwise_plain` | 0.928 | 0.906 | 72.3% |
| 合成データ | `jev_crossencode` | 0.840 | 0.741 | 64.7% |
| 合成データ | `decisions_crossencode` | 0.826 | 0.727 | 53.9% |

`decisions_crossencode` − `jev_pointwise`（95% 区間）:

| データ | 問題ごとの PR-AUC | 全問まとめ PR-AUC |
|---|---|---|
| MIRACL（797 問） | −0.067（−0.080〜−0.053） | −0.081（−0.097〜−0.064） |
| J-RAGBench（60 問） | −0.161（−0.218〜−0.111） | −0.160（−0.218〜−0.102） |
| 合成データ（60 問） | −0.126（−0.168〜−0.085） | −0.194（−0.249〜−0.138） |

## 読み方

測った値から言えることに限る。

- 1 件ずつの経路どうしでは、Decisions は MIRACL と J-RAGBench で Jev より低く、区間は 0 をまたがない。合成データでは差がはっきりしない（区間が 0 をまたぐ）
- Jev の最も成績のよい経路（10 件まとめる `jev_pointwise`）と比べると、3 つのデータすべてで Decisions が低く、区間は 0 をまたがない。Decisions は 10 件まとめる形を安定して読めなかったので、この形の上乗せを受けられない
- J-RAGBench で関連文書の 9 割を残すしきい値で切ると、残った文書のうち関連する割合は Decisions で 20.7% と低い。最良 F1 のしきい値も 0.01 で、関連文書の確率が低く出る組が多い（`judge` の画面の出力）
- ECE は MIRACL でだけ Decisions の方が小さい（0.133 と 0.160）。J-RAGBench と合成データでは Jev の 1 件ずつの経路の方が小さい
- 入力トークンは、同じ入力でも Decisions の方が 2 割ほど少ない（トークナイザーの違い）。料金は 1M トークンあたり Decisions $0.10、Jev $0.042（2026-10-19 の公式ページ）で、1 件ずつの経路の費用は Decisions が Jev の約 2 倍になった。Decisions のガイドにある long-context の料金乗数は 272K トークンを超えるリクエストにかかるもので、今回のリクエスト（1 件ずつで数百〜数千トークン）には関係しない
- 所要時間は、1 件ずつの経路で Jev と Decisions がほぼ同じだった（MIRACL で 388 秒と 392 秒、8 並列）。並列度、通信環境、時間帯で変わるので、速さの優劣は結論にしない

## 結果を読むときの注意

- データはすべて日本語。Decisions のガイドと `gpt-6-luna` のモデルページ（2026-10-19 取得）には、対応言語や日本語の精度についての記述が見当たらなかった
- Decisions の判定基準は指示文の文面に入れており、Jev の `criteria` と同じ渡し方ではない
- Decisions API は public beta で、GA までに挙動が変わりうる。今回の値は 2026-10-19 の `gpt-6-luna` のもの
- 合成データと MIRACL のラベルの注意点は、README の「結果を読むときの注意」と同じ
