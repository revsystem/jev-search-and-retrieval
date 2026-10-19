# Decisions API と Jev 1.14 の判定器比較の結果

2026-10-19〜20 に測定した。設計は `.claude/docs/specs/openai-decisions-comparison.md`、本測定の前の確認は `decisions-probe.md` にある。記事の下書きは qiita-articles の `public/newArticle043.md`。

## 条件

| | Jev | Decisions API |
|---|---|---|
| 答えたモデル（応答の `model` 欄） | `jev-1.14.0` | `gpt-6-luna` |
| 測った経路 | `jev_pointwise`、`jev_pointwise_plain`、`jev_crossencode` | `decisions_crossencode` |
| 並列度 | 8 | 8 |
| 試行回数とタイムアウト | 3 回、60 秒 | 3 回（SDK の `max_retries=2`）、60 秒 |

Decisions の 10 件まとめる 2 経路は、候補の順序を逆にすると確率が大きく動いたため、本測定から外した（`decisions-probe.md`）。データ、指示文、判定基準の文面、指標の計算は、記事（Jev と Cohere Rerank の比較）と同じ。Decisions の `predicate` には `criteria` の欄がないので、判定基準は指示文の末尾に入れた。

```bash
ROUTES=jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_crossencode
uv run jev-rag judge --dataset synthetic --rankers $ROUTES --out out/decisions/judge-synthetic.json
uv run jev-rag judge --dataset jragbench --rankers $ROUTES --out out/decisions/judge-jragbench.json
uv run jev-rag judge --dataset miracl --rankers $ROUTES --out out/decisions/judge-miracl.json
```

保存したスコアは `results/decisions/` に写した（組ごとの ID、正解ラベル、スコア、1 件ずつの経路の 4 段階の評価、経路ごとの利用量。文書の本文は含まない）。下の表と差の行は、すべて次のコマンドで再現できる。API も認証情報も要らない。

```bash
uv run jev-rag report-judge --results-dir results/decisions
```

## 測定の経緯

- 最初の MIRACL の測定では、`jev_crossencode` の 54 問が失敗した。その後の再実行が `402 billing_error`（TypeSafe のクレジット切れ）で止まったので、失敗もクレジット切れによると考えられる（応答本文は残っていないので未確認）。ユーザーがクレジットを追加して測り直し、失敗は 0 問になった
- レビューで、1 件ずつの経路の並べ替えの指標が、経路の並べ方と食い違っていると指摘された。経路は同じ確率の候補を 4 段階の評価で並べ分けるのに、指標は確率だけで計算していた。Decisions の確率は 0.00 に偏る（MIRACL 42%、J-RAGBench 82%、合成 44%）ので、Decisions に不利に出ていた。`score_pairs` が評価（`grade`）を保存するように直し、1 件ずつの 2 経路を 3 データで測り直した（2026-10-20）。Decisions は 1 回目と同じ確率と入力トークンを返した。教訓は `.claude/lessons/metrics.md`
- 測り直す前の結果は `out/decisions/before-grade/`（追跡対象外）にある

すべての経路で、拒否、欠けた答え、失敗した問題は 0 件だった。Jev のまとめる経路のリクエスト数は、3 つのデータとも「問題ごとの候補数を 10 件ずつに分けた数」と一致した（MIRACL 1,134、J-RAGBench 117、合成 60）。

## 結果

`report-judge --results-dir results/decisions` の出力:

| データ | 形 | 経路 | 答えたモデル | 問題ごとの PR-AUC | 全問まとめ PR-AUC | 9 割を残したときの関連の割合 | ECE | 拒否/欠け/失敗した問題 | 入力トークン | 費用（ドル） |
|---|---|---|---|---|---|---|---|---|---|---|
| MIRACL | 10 件ずつ、判定基準あり | `jev_pointwise` | jev-1.14.0 | 0.866 | 0.76 | 59.4% | 0.156 | 0/0/0 | 3,896,652 | 0.1637 |
| MIRACL | 1 件ずつ | `jev_crossencode` | jev-1.14.0 | 0.870 | 0.75 | 57.9% | 0.161 | 0/0/0 | 7,325,018 | 0.3077 |
| MIRACL | 1 件ずつ | `decisions_crossencode` | gpt-6-luna | 0.859 | 0.68 | 56.0% | 0.133 | 0/0/0 | 5,976,555 | 0.5977 |
| J-RAGBench | 10 件ずつ、判定基準あり | `jev_pointwise` | jev-1.14.0 | 0.938 | 0.87 | 62.0% | 0.051 | 0/0/0 | 552,982 | 0.0232 |
| J-RAGBench | 1 件ずつ | `jev_crossencode` | jev-1.14.0 | 0.920 | 0.78 | 53.5% | 0.066 | 0/0/0 | 855,195 | 0.0359 |
| J-RAGBench | 1 件ずつ | `decisions_crossencode` | gpt-6-luna | 0.911 | 0.71 | 20.7% | 0.112 | 0/0/0 | 701,465 | 0.0701 |
| 合成データ | 10 件ずつ、判定基準あり | `jev_pointwise` | jev-1.14.0 | 0.952 | 0.92 | 84.1% | 0.055 | 0/0/0 | 208,811 | 0.0088 |
| 合成データ | 1 件ずつ | `jev_crossencode` | jev-1.14.0 | 0.849 | 0.74 | 65.9% | 0.085 | 0/0/0 | 480,386 | 0.0202 |
| 合成データ | 1 件ずつ | `decisions_crossencode` | gpt-6-luna | 0.845 | 0.73 | 53.9% | 0.157 | 0/0/0 | 397,627 | 0.0398 |

Decisions − Jev（問題を単位に 2,000 回の bootstrap、95% 区間）:

| データ | 比べた形 | 比べた問題 | 問題ごとの PR-AUC | 全問まとめ PR-AUC |
|---|---|---|---|---|
| MIRACL | 1 件ずつどうし | 797 問 | −0.011（−0.023〜+0.001） | −0.07（−0.08〜−0.05） |
| MIRACL | Jev は 10 件ずつ、Decisions は 1 件ずつ | 797 問 | −0.007（−0.020〜+0.005） | −0.08（−0.10〜−0.06） |
| J-RAGBench | 1 件ずつどうし | 60 問 | −0.009（−0.038〜+0.018） | −0.07（−0.14〜−0.02） |
| J-RAGBench | Jev は 10 件ずつ、Decisions は 1 件ずつ | 60 問 | −0.027（−0.054〜−0.005） | −0.16（−0.22〜−0.10） |
| 合成データ | 1 件ずつどうし | 60 問 | −0.004（−0.034〜+0.027） | −0.01（−0.05〜+0.02） |
| 合成データ | Jev は 10 件ずつ、Decisions は 1 件ずつ | 60 問 | −0.107（−0.146〜−0.070） | −0.19（−0.25〜−0.14） |

上位の並び（report-judge の同じ節の表）、でたらめに並べた場合の問題ごとの PR-AUC（MIRACL 0.37、J-RAGBench 0.36、合成 0.47、2,000 回の無作為な並べ替えの平均）も report-judge に出る。

## 読み方

測った値から言えることに限る。

- 1 件ずつどうしでは、問題ごとの PR-AUC の差は 3 つのデータすべてで区間が 0 をまたぎ、並べ替える力に差ははっきりしない
- 確率だけで並べると、Decisions の問題ごとの PR-AUC は MIRACL 0.799、J-RAGBench 0.777、合成 0.826 まで下がる。`predicate` の確率だけを使う実装では、この値が実際の成績になる
- 全問まとめた PR-AUC では、MIRACL と J-RAGBench で Decisions が下回り、区間は 0 をまたがない。1 つのしきい値ですべての質問をふるい分ける使い方では差が出る
- J-RAGBench の Decisions は、関連文書 147 件のうち 45 件に 0.00 を付けた。関連文書の 9 割を残すしきい値は 0.00 になり、すべての文書が残るので、関連の割合は関連文書の割合と同じ 20.7% になる
- `jev_pointwise` と比べると、J-RAGBench と合成データで Decisions が下回る（区間は 0 をまたがない）。合成データの差の大部分は、Jev が 10 件まとめると上がる分
- 1 件ずつの費用は Decisions が Jev の約 2 倍。入力トークンは同じ文面でも Decisions の方が 2 割ほど少ないが、単価が 2 倍以上
- Jev の 10 件ずつが 1 件ずつより安い理由は、合成データの 1 問（`a-011`）で測った。10 件ずつ 1 リクエスト 3,537 トークン、1 件ずつ 10 リクエスト合計 7,944 トークン。4 段階の評価の質問が 1 問 173 トークン、ほぼ空のリクエストでも 271 トークン（リクエストごとの固定分と考えられる。ドキュメントに説明はない）
- 所要時間（1 件ずつ、MIRACL、8 並列）は Jev 457 秒、Decisions 440 秒。速さの優劣は結論にしない
- 確率も評価も同じ候補は受け取った順に並ぶ。Decisions の MIRACL で、1 位がこの状態の問題は 797 問中 50 問

## 結果を読むときの注意

- データはすべて日本語。Decisions のガイドと `gpt-6-luna` のモデルページ（2026-10-19 取得）には、対応言語や日本語の精度の記述が見当たらなかった。Jev は英語が主な学習言語と公式ドキュメントにある
- Decisions の判定基準は指示文の文面に入れており、Jev の `criteria` と同じ渡し方ではない
- Decisions API は public beta で、GA までに挙動が変わりうる
- Decisions のガイドにある long-context の料金乗数は、モデルページでは 272K トークンを超えるリクエストにかかるもので、今回のリクエストには関係しない
