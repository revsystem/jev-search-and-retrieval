# 候補の順序の入れ替え確認（Decisions API と Jev 1.14）

2026-10-19 01:45 ごろに実行した。判定の基準と目的は `.claude/docs/specs/openai-decisions-comparison.md` の「本測定の前の確認」にある。

## 実行したこと

```bash
uv run python out/smoke_decisions.py   # 各リクエストの形を 1 回ずつ送る疎通確認
uv run jev-rag probe-order             # 合成データ 5 問（50 組）を、そのままの順と逆順で送る
```

疎通確認では、10 件まとめた `predicate` と、1 件ずつの `predicate` と `score`（4 段階）の組が、どちらもエラーなしで受理された。答えたモデルは `gpt-6-luna` と `jev-1.14.0`、拒否と欠けた答えは 0 件だった。

最初の疎通確認では、API キーに権限がなく `403 The API key requires the api.decisions.write permission.` を受け取った。ユーザーがキーに権限を付けてから再実行した。

## 結果

`probe-order` の出力:

| 経路 | 組 | 差の中央値 | スピアマン | 文書に付いてくる | 問題ごとの PR-AUC | 失敗 |
|---|---|---|---|---|---|---|
| decisions_pointwise | 50 | 0.030 | 0.587 | いいえ | 0.918 | 0 |
| decisions_crossencode | 50 | 0.000 | 1.000 | はい | 0.908 | 0 |
| jev_pointwise | 50 | 0.030 | 0.935 | はい | 1.000 | 0 |
| jev_crossencode | 50 | 0.010 | 0.995 | はい | 0.947 | 0 |

まとめる経路と 1 件ずつの経路の問題ごとの PR-AUC の差は、Decisions も Jev も 0.2 未満だった。

保存した結果（`out/decisions/probe-order.json`）から追加で計算した値:

| 経路 | 問題ごとの PR-AUC（順方向 / 逆順） | 差が 0.2 を超えた組 | 両方とも 0.05 未満の組を除いたスピアマン |
|---|---|---|---|
| decisions_pointwise | 0.918 / 0.838 | 16 / 50 | 0.108（25 組） |
| decisions_crossencode | 0.908 / 0.908 | - | - |
| jev_pointwise | 1.000 / 0.983 | 2 / 50 | 0.914（42 組） |
| jev_crossencode | 0.947 / 0.879 | - | - |

## 読み方

1 件ずつの経路は、順序を逆にしても同じ入力になるので、2 回の差は実行ごとの揺れだけを表す。`decisions_crossencode` は差 0.000、スピアマン 1.000 で、Decisions API は同じ入力に同じ確率を返した。`jev_crossencode` も差の中央値 0.010、スピアマン 0.995 で基準を満たした。基準は揺れではなく順序の影響を測れている。

`decisions_pointwise` は、差の中央値は小さいが、スピアマンが基準の 0.8 を下回った。0 付近の小さな揺れのせいではない。両方とも 0.05 未満の組を除くとスピアマンは 0.108 まで下がり、50 組のうち 16 組は順序を逆にしただけで確率が 0.2 以上動いた（例: 無関連の `b-027-d07` が 0.87 から 0.00、関連の `a-025-d01` が 0.77 から 0.05）。Decisions API は同じ入力に同じ値を返すので、この動きは候補の並び順によるもので、`candidates[i].text` のパスで候補を指す形を安定して読めていない。

`jev_pointwise` は基準を満たした。Jev 1.14 では、10 件まとめた経路が引き続き成り立つ。

5 問だけの問題ごとの PR-AUC は揺れに弱い。`jev_crossencode` は入力が同じでも 0.947 と 0.879 で、0.07 動いた。「差が 0.2 未満」の判定は、この規模ではおおまかな目安にとどまる。

## 本測定の経路

仕様の基準に従い、`decisions_pointwise` と `decisions_pointwise_plain` は本測定から外し、Decisions は `decisions_crossencode` だけを測る。Jev は 3 経路とも測る。

```
jev_pointwise,jev_pointwise_plain,jev_crossencode,decisions_crossencode
```
