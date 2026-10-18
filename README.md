# strands-decider は RAG の rerank で関連文書を正しく見分けられるか

Strands Agents の判定専用モデル strands-decider（Strands Decider 2B）を、RAG の rerank の判定に使ったとき、TypeSafe の Jev と比べて関連文書をどれだけ正しく見分けられるかを測るブランチです。rerank は、検索で集めた候補文書を、質問への関連度で並べ替える工程です。strands-decider 0.1.0 のサーバーを手元の GPU で動かし、v19 のモデル（`StrandsAgents/strands-decider-2B-hobson-v19`）に、Jev と同じ質問文と判定基準で、同じ質問と候補文書の組を判定させて比べます。

このブランチのコマンドは、Qiita の記事「[AWS がリリースした判定専用モデル「Strands Decider 2B」を Jev と同じコーパスで検証した](https://qiita.com/revsystem/items/6a00d222dee59c7dec4a)」の検証を再現するためのものです。記事の執筆時に測定した結果は `results/` に入っており、記事と下の「結果」の表はこの結果から集計しています。

データ、成績の測り方、Jev と Cohere Rerank の測定は、前回の記事「[TypeSafe Jev を RAG の rerank に使うと、Cohere Rerank より関連文書を正しく見分けられるのか検証した](https://qiita.com/revsystem/items/b619a211781af138e738)」と同じです。前回の記事の結果は、デフォルトブランチの [README](https://github.com/revsystem/jev-search-and-retrieval#readme) にあります。前回の記事の再現手順を変えないよう、strands-decider 用のコードはこのブランチに置いています。

## 比べる手法

どの手法も、データセットに含まれる同じ候補文書にスコアを付けます。strands-decider と Jev には、判定させる材料（質問と候補文書。API では state と呼ぶ）と、「この候補文書は質問に答えるための根拠として役に立ちますか」という問いを送り、「はい」の確率を返させます。strands-decider の経路は `src/jev_rag/decider.py` にあり、Jev と同じ `POST /v1/systemone` の形でローカルのサーバーに送ります。

| 経路 | 中身 |
|---|---|
| `decider_single` | strands-decider。候補文書 1 件ずつ、Jev と同じ日本語の問いと判定基準（何を「はい」とするかの説明）で尋ねる |
| `decider_single_plain` | strands-decider。1 件ずつで、判定基準を渡さず「関連しますか」とだけ尋ねる。0.1.0 のサーバーは、判定基準を省くと英語の汎用の基準で補う |
| `decider_single_en` | strands-decider。1 件ずつで、問いと判定基準を英語にする（候補文書は日本語のまま） |
| `decider_pointwise` | strands-decider。候補文書を最大 10 件まとめて送り、`candidates[0].text` のように 1 件ずつ指して尋ねる（`jev_pointwise` と同じ形） |
| `jev_pointwise` | Jev。1 つの質問の候補文書を 10 件ずつまとめて送る |
| `jev_crossencode` | Jev。候補文書 1 件ずつ送る |
| `cohere_rerank` | Amazon Bedrock の Cohere Rerank 3.5 |

`results/` には、判定基準なしと英語を最大 10 件ずつで測った `decider_pointwise_plain`、`decider_pointwise_en` のスコアも入っています。記事と `report-judge` では扱っていません。

strands-decider 0.1.0 が一度に読める入力は 4,096 トークン（モデルが文章を数える単位）までで、超えた分はエラーにならずに切り捨てられます。`decider.py` は、strands-decider と同じ数え方（`Qwen/Qwen3.5-2B-Base` のトークナイザー）で送る材料の長さを数え、問いの分を残して 3,600 トークン以内に収まるように候補文書を区切ってから送ります。そのため、候補文書が長いデータでは、`decider_pointwise` の 1 回に送る件数が 10 件未満になります。1 件ずつなら、すべての候補文書が上限に収まります。同時にリクエストを送ったときの動作は公式に検証されていないので、リクエストは 1 件ずつ順に送ります。

## データ

性質の違う 3 つのデータを、混ぜずに別々に集計します。

| データ | 中身 | 規模 | 関連文書の割合 |
|---|---|---|---|
| MIRACL 日本語 dev | Wikipedia の検索結果に人が関連の有無を付けた公開データ。質問は一問一答型 | 860 問、8,354 組 | 21% |
| J-RAGBench | 架空企業の業務寄りの質問。回答不能な 54 問は関連文書がないので除く | 60 問、709 組 | 21% |
| 合成データ | 比較・理由・経緯・複数要因・統合の 5 種類の業務の質問。同じ話題で必要な事実だけがない紛らわしい無関係文書を含む | 60 問、600 組 | 32% |

合成データの仕様は `.claude/docs/specs/synthetic-judge-dataset.md` にあり、問題と文書は `data/synthetic/` に入っています。作り方と確かめ方は、デフォルトブランチの README を参照してください。

## 使い方

`results/` に入っている記事における測定結果から、下の「結果」の表と、記事の結果1と結果2の表を再現できます。この場合は strands-decider のサーバーも、API も認証情報も要りません。

```bash
git clone https://github.com/revsystem/jev-search-and-retrieval.git
cd jev-search-and-retrieval
git switch feat/strands-decider-comparison
uv venv
uv pip install -e ".[dev]"
uv run jev-rag report-judge --results-dir results
```

`report-judge` は、前回の記事の表に続けて「strands-decider との比較」の表を出し、「上位に関連文書が来るか」の表に strands-decider の行を加えます。差に付ける 95% 区間は、問題を選び直して差を計算し直す操作を 2,000 回繰り返し、上下の 2.5% ずつを除いた範囲です。区間の両端がプラスなら、問題の選び方が変わっても差の向きは変わりにくいと読めます。選び直しの乱数は固定（`--seed 0`）なので、同じスコアからは同じ値が出ます。

### 測定をやり直す

strands-decider のサーバーは、このリポジトリとは別の仮想環境に入れて起動します。NVIDIA の GPU で動かすときに追加で入れる flash-linear-attention と causal-conv1d、WSL2 での CUDA ツールキットの入れ方は、記事の「検証環境」の節にあります。`decider.py` は送り先を `http://127.0.0.1:8099/v1`、モデルを `strands-decider-2B-hobson-v19` に固定しているので、ポート 8099 で v19 のモデルを起動します。

```bash
# strands-decider 用の仮想環境で
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cuda --port 8099
```

サーバーを起動したら、このリポジトリで測定します。`judge` は、始める前にサーバーの `/health` を読み、モデル名、デバイス、入力の上限を表示します。初回は、長さを数えるために Hugging Face から `Qwen/Qwen3.5-2B-Base` のトークナイザーを取得します。

```bash
uv run jev-rag prepare-judge  # J-RAGBench と MIRACL 日本語版を取得する（MIRACL のコーパスは約 1GB）

uv run jev-rag judge --dataset miracl --rankers decider_single,decider_single_plain,decider_single_en,decider_pointwise
uv run jev-rag judge --dataset jragbench --rankers decider_single,decider_single_plain,decider_single_en,decider_pointwise
uv run jev-rag judge --dataset synthetic --rankers decider_single,decider_single_plain,decider_single_en,decider_pointwise

uv run jev-rag report-judge  # out/ に保存した結果を集計する
```

strands-decider の経路だけなら、`.env` の Jev の API キーも AWS の認証も要りません。`judge` は、全経路のスコアを組ごとに `out/judge-<dataset>.json` へ保存し、測定済みの経路は再実行しても呼び直しません。strands-decider の経路だけを測ると、`report-judge` の表には Cohere Rerank と Jev の行が出ません。並べて比べるには、前回の記事の経路（`judge` の既定の経路）も同じ `out/` に測っておきます。手順と `.env` の書き方は、デフォルトブランチの README にあります。

## 結果

成績は PR-AUC で示します。関連文書を上位に、無関係文書を下位に並べられているほど 1 に近づく 0〜1 の値です。下の表は、記事における測定結果（`results/`）を `report-judge` で集計した値です。

### 1 つの質問に対する候補文書を並べ替える力

質問ごとに PR-AUC を計算して平均した値です。MIRACL は、候補文書がすべて関連文書の 63 問を除いた 797 問で計算しています。

| データ | Cohere Rerank | Jev（1 件ずつ） | Jev（10 件ずつ） | strands-decider（1 件ずつ） | strands-decider（最大 10 件ずつ） |
|---|---|---|---|---|---|
| MIRACL（797 問） | 0.841 | 0.860 | 0.865 | 0.692 | 0.412 |
| J-RAGBench（60 問） | 0.932 | 0.909 | 0.931 | 0.818 | 0.496 |
| 合成データ（60 問） | 0.521 | 0.867 | 0.940 | 0.785 | 0.489 |

Jev と strands-decider（1 件ずつ）の差です。渡し方が同じ「1 件ずつ」どうしの差と、それぞれの一番良い渡し方（Jev は 10 件ずつ）どうしの差を並べます。

| データ | Jev（1 件ずつ）− strands-decider（1 件ずつ） | Jev（10 件ずつ）− strands-decider（1 件ずつ） |
|---|---|---|
| MIRACL | +0.168（+0.146〜+0.188） | +0.173（+0.152〜+0.194） |
| J-RAGBench | +0.091（+0.043〜+0.145） | +0.113（+0.070〜+0.162） |
| 合成データ | +0.082（+0.033〜+0.126） | +0.155（+0.110〜+0.203） |

3 つのデータすべてで Jev が上回り、どちらの比べ方でも区間の両端はプラスでした。1 件ずつの strands-decider は、Cohere Rerank と比べると、合成データでは上回り、MIRACL と J-RAGBench では下回りました。

候補文書を最大 10 件まとめて渡すと、でたらめに並べた場合（MIRACL 0.37、J-RAGBench 0.36、合成データ 0.47）に近い成績になります。strands-decider 0.1.0 のサーバーは、送った材料を JSON の文字列にしてそのまま読むだけで、問いの中の `candidates[1].text` のような書き方で候補文書を指す仕組みがありません。そのため、どの候補文書について問われているかを区別できなかったと考えられます。

### 判定基準のありなしと、問いの言語

strands-decider の 1 件ずつの経路で比べた、質問ごとの PR-AUC の平均です。どれも差は小さく、判定基準や問いの言語を変えても成績はほとんど変わりませんでした。

| データ | 判定基準あり（日本語） `decider_single` | 判定基準なし `decider_single_plain` | 判定基準あり（英語） `decider_single_en` |
|---|---|---|---|
| MIRACL | 0.692 | 0.691 | 0.694 |
| J-RAGBench | 0.818 | 0.836 | 0.838 |
| 合成データ | 0.785 | 0.756 | 0.788 |

### 確率の当たり具合と処理時間

| データ | 経路 | ECE | 1 リクエストの処理 ms（中央値） | 1 問あたりの秒 |
|---|---|---|---|---|
| MIRACL | Jev（10 件ずつ） | 0.136 | - | 0.33 |
| MIRACL | strands-decider（1 件ずつ） | 0.464 | 82 | 0.88 |
| J-RAGBench | Jev（10 件ずつ） | 0.049 | - | 0.33 |
| J-RAGBench | strands-decider（1 件ずつ） | 0.224 | 124 | 1.55 |
| 合成データ | Jev（10 件ずつ） | 0.083 | - | 0.29 |
| 合成データ | strands-decider（1 件ずつ） | 0.259 | 80 | 0.82 |

ECE は、返した確率が当たっているかを見る値です。たとえば 0.7 と答えた文書のうち実際に約 7 割が関連していれば当たっていて、そのずれが小さいほど 0 に近づきます。strands-decider の確率は「関連する」の側に大きく寄ります。1 リクエストの処理時間は、サーバーが返す処理時間（候補文書 1 件、RTX 3060 Ti）の中央値です。1 問あたりの秒は実際にかかった時間で、strands-decider は手元のサーバーに 1 件ずつ順に送り、Jev はネットワーク越しに 10 件ずつ送っています。送り方が違うので、速さは比べていません。

## 結果を読むときの注意

- 測ったのは、v19 のモデルを strands-decider 0.1.0 で、RTX 3060 Ti の 1 枚で動かした場合だけ。処理時間はこの環境での値
- 結果の表は causal-conv1d を入れる前の値。入れたあとに 1 件ずつの経路を合成データと MIRACL で測り直したが、処理時間にもスコアにもほとんど違いはなかった（値は記事の「causal-conv1d のビルド」の節）。測り直したスコアは `results/` に入れていないので、このリポジトリからは再現できない
- Jev と Cohere Rerank のスコアは前回の記事の測定のもので、strands-decider と同じ日に測り直してはいない
- データはすべて日本語。strands-decider のドキュメントとモデルカードには、対応する言語の記述がない
- 1 件ずつの経路の PR-AUC は、別々のリクエストで出た確率をそのまま並べて計算している。strands-decider でこの並べ方が妥当かは確かめていない

## 構成

```
results/        記事における測定結果（組ごとの ID、正解ラベル、スコア。文書の本文は含まない）
src/jev_rag/
  decider.py    strands-decider の経路（入力の上限に収まるよう候補文書を区切り、手元のサーバーに 1 件ずつ順に送る）
  judge.py      組ごとの採点、PR-AUC などの指標、区間の計算、judge と dilution の実行
  judge_report.py  保存済みのスコアから記事の表をすべて計算する（report-judge）
  public.py     J-RAGBench と MIRACL 日本語版の取得と読み込み
  synthetic.py  合成データの読み込み、検証用ビュー、突き合わせ、素朴な判定
  rankers.py    比較する経路
  bedrock.py    Cohere Embed v4 / Cohere Rerank 3.5 / GPT-5.6 Luna
  jev/          Jev クライアントと、公式ユースケースごとの実装
  cli.py        jev-rag コマンド
```

## 出典とライセンス

- [strands-decider](https://github.com/strands-labs/strands-decider)、[Introducing Strands Decider（Strands Agents の公式ブログ）](https://strandsagents.com/blog/introducing-strands-decider/)
- [MIRACL](https://huggingface.co/datasets/miracl/miracl)（miracl/miracl、miracl/miracl-corpus）: Apache-2.0
- [J-RAGBench](https://huggingface.co/datasets/neoai-inc/Japanese-RAG-Generator-Benchmark)（neoai-inc）: CC BY-SA 4.0。スコアの掲載は出典表示で足り、データや改変版を再配布するときは同じライセンスを引き継ぐ
- [TypeSafe AI ドキュメント](https://docs.typesafe.ai/)
- [Amazon Bedrock の Rerank](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html)
