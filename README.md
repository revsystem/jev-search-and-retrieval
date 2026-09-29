# Jev は RAG の rerank で関連文書を正しく見分けられるか

TypeSafe の System One モデル Jev を、RAG の rerank（検索で集めた候補文書を、質問への関連度で並べ替える工程）の判定器として使ったとき、Cohere Rerank より関連文書を正しく見分けられるかを測るリポジトリです。質問と候補文書の組ごとに各手法のスコアを付け、「関連する／しない」をどれだけ正しく見分けたかを比べます。

測定結果の詳細は `.claude/docs/research/judge-results.md` にあります。

## 比べる手法

どの手法も、データセットに含まれる同じ候補文書にスコアを付けます。埋め込みもベクトル索引から検索するのではなく、質問と各候補文書のベクトルの近さをそのままスコアにします。

| 経路 | 中身 |
|---|---|
| `length` / `term_overlap` / `bm25` | 文書の長さ、質問との語の重なり、キーワード一致（BM25）。データに手がかりが漏れていないかを見る素朴な判定器 |
| `embedding` | Cohere Embed v4 のベクトルのコサイン類似度 |
| `cohere_rerank` | Amazon Bedrock の Cohere Rerank 3.5 |
| `jev_pointwise` | Jev。1 つの質問の候補文書を 10 件ずつ 1 つの state にまとめ、判定基準（`criteria`）付きの Noul で「根拠として役に立つか」の確率を返させる |
| `jev_pointwise_plain` | Jev。判定基準を渡さず「関連しますか」とだけ尋ねる版 |
| `jev_crossencode` | Jev。候補文書 1 件ごとに 1 リクエスト |

## データ

性質の違う 3 つのデータを、混ぜずに別々に集計します。

| データ | 中身 | 規模 | 関連文書の割合 |
|---|---|---|---|
| MIRACL 日本語 dev | Wikipedia の検索結果に人が関連の有無を付けた公開データ。質問は一問一答型 | 860 問、8,354 組 | 21% |
| J-RAGBench | 架空企業の業務寄りの質問。回答不能な 54 問は関連文書がないので除く | 60 問、709 組 | 21% |
| 合成データ | 比較・理由・経緯・複数要因・統合の 5 種類の業務の質問。同じ話題で必要な事実だけがない紛らわしい無関係文書を含む | 60 問、600 組 | 32% |

合成データの仕様は `.claude/docs/specs/synthetic-judge-dataset.md` にあります。正解は作り方で決め（必要な事実を書いた文書が関連）、後から AI に判定させて決めてはいません。生成は Claude（Opus 5.5 が 3 問、Sonnet 5 が 27 問）と Grok 4.7 が 30 問ずつ、検証は作っていない側（Cursor Agent と Claude Sonnet 5）が正解を伏せたビューで行い、作成時の正解と全文書で一致しました。無作為に選んだ 6 問は人が目で確認しています。

| パス | 中身 |
|---|---|
| `data/synthetic/gen_a/`、`gen_b/` | 問題と文書、作成時の正解（`role`、`contains_facts`） |
| `data/synthetic/views_a/`、`views_b/` | 正解を取り除き、文書の順序を入れ替えた検証用ビュー |
| `data/synthetic/verify_a/`、`verify_b/` | 検証担当の判定 |

## 使い方

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env

aws sso login --profile <プロファイル名>  # Bedrock 用。鍵ではなく SSO
uv run jev-rag check                   # Jev の疎通確認
uv run jev-rag prepare-judge           # J-RAGBench と MIRACL 日本語版を取得する（MIRACL のコーパスは約 1GB）

# 合成データの関門（有料の測定の前に通す）
uv run jev-rag synthetic-check --gen data/synthetic/gen_a --verify data/synthetic/verify_a
uv run jev-rag synthetic-check --gen data/synthetic/gen_b --verify data/synthetic/verify_b

# 測定（API を呼ぶ。組ごとのスコアを out/ に保存する）
uv run jev-rag judge --dataset miracl
uv run jev-rag judge --dataset jragbench
uv run jev-rag judge --dataset synthetic
uv run jev-rag dilution

# 集計（API を呼ばない。保存したスコアから下の「結果」の表をすべて出す）
uv run jev-rag report-judge                        # 自分で測った out/ を集計する
uv run jev-rag report-judge --results-dir results  # 私たちの測定結果（results/）を集計する
```

測定をやり直さなくても、`results/` に入っている私たちの測定結果から、下の「結果」の表を再現できます。この場合は API も認証情報も要りません。

```bash
uv venv
uv pip install -e ".[dev]"
uv run jev-rag report-judge --results-dir results
```

`synthetic-check` は、素朴な判定器（文書の長さ、語の重なり、BM25）の成績と、作成時の正解と検証結果の突き合わせを表示します。食い違いか未検証の問題があれば、終了コード 1 を返します。

`judge` は、全経路のスコアを組ごとに `out/judge-<dataset>.json` へ保存します。既定の経路は `embedding,cohere_rerank,jev_pointwise,jev_pointwise_plain,jev_crossencode` で、素朴な判定器は毎回あわせて計算します。測定済みの経路は再実行しても呼び直さないので、途中で止まっても続きから再開できます。画面には全問まとめの PR-AUC などを表示します。

`dilution` は、1 リクエストに載せる件数の測定結果を `out/dilution.json` に保存します。

`report-judge` は、この 2 種類のファイルから下の「結果」の表をすべて出します。差の区間は、問題を重複を許して選び直す操作を 2,000 回繰り返したときの 2.5〜97.5 パーセンタイルです。乱数の種は固定（`--seed 0`）なので、同じスコアからは同じ値が出ます。1 回の実行に 1〜2 分かかります。

API の応答は実行ごとに少し揺れるので、`judge` と `dilution` を測り直すと、値は下の表と完全には一致しません。下の表は、`results/` の測定結果を `report-judge` で集計した値です。

`.env` に書く秘密情報は Jev の API キー（`TYPESAFE_API_KEY`、Vercel AI Gateway 経由なら `AI_GATEWAY_API_KEY`）です。Bedrock の認証情報は書かず、`AWS_PROFILE` にプロファイル名を指定して `aws sso login` で認証します。

## 結果

PR-AUC は、関連文書を上位に、無関係文書を下位に並べられているほど 1 に近づく 0〜1 の値です。どの表も、3 つのデータの `judge` の結果を `report-judge` で集計したものです。

### 1 つの質問に対する候補文書を並べ替える力

問題ごとに PR-AUC を計算して平均した値です。rerank に必要なのはこちらです。`report-judge` の同じ見出しの表に対応します（`judge` の画面には出ません）。MIRACL は、候補文書がすべて関連文書の 63 問を除いた 797 問で計算しています。

| データ | でたらめ | 埋め込み | Cohere Rerank | Jev | Jev − Cohere Rerank（95% 区間） |
|---|---|---|---|---|---|
| MIRACL（797 問） | 0.37 | 0.771 | 0.841 | 0.865 | +0.024（+0.007〜+0.040） |
| J-RAGBench（60 問） | 0.36 | 0.880 | 0.932 | 0.931 | −0.001（−0.045〜+0.043） |
| 合成データ（60 問） | 0.47 | 0.487 | 0.521 | 0.940 | +0.419（+0.367〜+0.465） |

### 質問をまたいでスコアの意味がそろっているか

全問題の組をまとめて 1 本に並べて計算した PR-AUC です。1 つのしきい値で全質問を切る使い方に必要なのはこちらです。`report-judge` の同じ見出しの表に対応します。区間以外の値は、`judge` の画面の「PR-AUC」の列にも出ます。

| データ | でたらめ | キーワード一致（BM25） | 埋め込み | Cohere Rerank | Jev | Jev − Cohere Rerank（95% 区間） |
|---|---|---|---|---|---|---|
| MIRACL | 0.21 | 0.29 | 0.54 | 0.69 | 0.76 | +0.07（+0.04〜+0.09） |
| J-RAGBench | 0.21 | 0.64 | 0.70 | 0.79 | 0.88 | +0.09（+0.02〜+0.16） |
| 合成データ | 0.32 | 0.35 | 0.33 | 0.36 | 0.92 | +0.56（+0.52〜+0.59） |

読み方は次のとおりです。

- 一問一答に近い検索（MIRACL、J-RAGBench）では、並べ替えの精度は Cohere Rerank とほぼ同じ
- 同じ話題で中身だけが違う文書が並ぶ合成データでは、Cohere Rerank と埋め込みは当て推量の水準まで落ち、Jev との差が大きく開く。ただしこの差は合成データでしか確かめていない
- 1 つのしきい値で全質問を切る使い方では、Jev の方が無関係な文書を多く落とせる。関連文書の 9 割を残すしきい値で切ると、MIRACL で残った文書のうち関連する割合は Cohere Rerank 42.2%、Jev 57.7%（+15.5 ポイント、+12.4〜+19.3）。`report-judge` の「1 つのしきい値で全質問を切ったとき」の表に対応する
- 判定基準を渡さなくても、合成データで Jev は Cohere Rerank を大きく上回る（0.922 と 0.521）。差の大半はモデル自体から来ている。`report-judge` の「判定基準のありなし」の表に対応する
- 候補文書 10 件の rerank 1 回の費用は、Jev が約 0.0002 ドル、Cohere Rerank が約 0.002 ドル（料金表からの計算）

`report-judge` は、このほかに上位 k 件の表（1 位が関連文書である問題の割合など）も出します。

### 1 リクエストに載せる件数

合成データの各問題の 10 件に、他の問題の文書 90 件を混ぜて測りました。値は問題ごとの PR-AUC の平均です。`dilution` の結果を `report-judge` で集計した「1 リクエストに載せる件数」の表に対応します。

| 候補文書の中身 | 1 リクエストの件数 | PR-AUC |
|---|---|---|
| 同じ問題の 10 件 | 1 件ずつ | 0.87 |
| 同じ問題の 10 件 | 10 件まとめて | 0.94 |
| 混ぜた 100 件 | 10 件ずつ | 0.84 |
| 混ぜた 100 件 | 25 件ずつ | 0.83 |
| 混ぜた 100 件 | 50 件ずつ | 0.72 |
| 混ぜた 100 件 | 100 件まとめて | 0.55 |

同じ質問の候補文書を 10 件まとめて渡すと、1 件ずつより正確になります（+0.076、+0.046〜+0.108）。一緒に並ぶのが無関係な文書だと上乗せは消え、50 件を超えると崩れます。効いているのは件数ではなく、一緒に並ぶ中身です。

### 検索そのものを Jev に任せる場合

精度は測っていません。全文書に関連の有無が付いたデータがないためです。所要時間と費用の見積もり（文書を 10 件ずつ 1 リクエストにまとめ、8 並列で送った実測から）は次のとおりです。API の上限は 1 分あたり 1,200 リクエストです。

| 文書数 | 1 クエリのリクエスト数 | 所要時間（8 並列） | API 上限での最短 | 1 クエリの費用 |
|---|---|---|---|---|
| 1,000 | 100 | 14〜31 秒 | 5 秒 | 約 0.02 ドル |
| 10,000 | 1,000 | 2.4〜5 分 | 50 秒 | 約 0.2 ドル |
| 1,000,000 | 100,000 | 4〜8.5 時間 | 約 83 分 | 約 20 ドル |

## 結果を読むときの注意

- データはすべて日本語。公式ドキュメントの Models は、Jev の主な学習言語は英語で、他言語は英語ほど精度が高くないと書いている
- MIRACL のラベルには漏れがある。Jev が各問題で 1 位に置いた文書のうち、MIRACL のラベルでは「無関係」と付いていたもの（127 件）から 10 件を選び、ラベルを伏せて 2 つの AI に判定させると、5 件は両方が「答えが書かれている」と判定した（`.claude/docs/research/miracl-blind-check/`）。全体でどの程度あるかは確かめていない
- 合成データは LLM が書いた文章で、LLM の書いた文章を AI モデルが判定するときの偏りは確かめていない
- 所要時間は並列度、通信環境、測定した時間帯で変わる。速さの優劣は結論にしていない

## Jev の 2 経路

同じリクエストボディを 2 つの宛先に送り分けます。`JevClient` から上のコードは経路を知りません。`JEV_TRANSPORT=direct` と `TYPESAFE_API_KEY` で `https://api.typesafe.ai/v1/systemone` に `jev-latest` として送り、`JEV_TRANSPORT=gateway` と `AI_GATEWAY_API_KEY` で Vercel AI Gateway 経由（`typesafe-ai/jev`）で送ります。測定はすべて direct 経路で行いました。

入力の上限は、state と最長の質問 1 つで 32k トークン、state と全質問で 64k トークンです。上限を超えそうなときは候補文書を自動で分割します。

## 構成

```
results/        私たちの測定結果（組ごとの ID、正解ラベル、スコア。文書の本文は含まない）
src/jev_rag/
  judge.py      組ごとの採点、PR-AUC などの指標、bootstrap 区間、judge と dilution の実行
  judge_report.py  保存済みのスコアから記事の表をすべて計算する（report-judge）
  public.py     J-RAGBench と MIRACL 日本語版の取得と読み込み
  synthetic.py  合成データの読み込み、検証用ビュー、突き合わせ、素朴な判定器
  rankers.py    比較する経路
  bedrock.py    Cohere Embed v4 / Cohere Rerank 3.5 / GPT-5.6 Luna
  jev/          Jev クライアントと、公式ユースケースごとの実装
  cli.py        jev-rag コマンド
```

`jev/` の中身は公式ドキュメントと照合済みで、取得した公式ページは `.claude/docs/research/typesafe/` に保存してあります。`uv run jev-rag usecases` で、公式の Search and retrieval ユースケースと実装の対応を表示できます。

## 旧測定（JQaRA）

最初は JQaRA（クイズ形式の日本語検索データ）で nDCG@10 を測っていました。一般的な RAG の、一問一答では終わらない質問を代表しないため、判定器としての評価に切り替えました。`prepare`、`evaluate`、`answer`、`sweep`、`report`、`context` はこの旧測定のコマンドで、動作はしますが、上の結果には使っていません。旧測定の計画は `.claude/docs/archive/jqara-comparison.md` にあります。

## 出典とライセンス

- [MIRACL](https://huggingface.co/datasets/miracl/miracl)（miracl/miracl、miracl/miracl-corpus）: Apache-2.0
- [J-RAGBench](https://huggingface.co/datasets/neoai-inc/Japanese-RAG-Generator-Benchmark)（neoai-inc）: CC BY-SA 4.0。スコアの掲載は出典表示で足り、データや改変版を再配布するときは同じライセンスを引き継ぐ
- [JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA)（旧測定）: question と answers は JAQKET 由来で CC-BY-SA-4.0、passage は Wikipedia の CC BY-SA 4.0 または GFDL
- [TypeSafe AI ドキュメント](https://docs.typesafe.ai/)、[JEV-as-a-Judge（arXiv:2609.26550）](https://arxiv.org/abs/2609.26550)
- [Amazon Bedrock の Rerank](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html)、[Amazon Bedrock の料金](https://aws.amazon.com/bedrock/pricing/)
