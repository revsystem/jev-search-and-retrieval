# Jev は RAG の検索精度を上げるのか

TypeSafe の System One モデル Jev を検索・再ランキングに挟むと、従来のRAGと比べて実際にスコアが上がるのか。公式ドキュメントの Search and retrieval ユースケースごとに実装し、同じ問題・同じ候補・同じ指標で横並びに測るためのリポジトリです。

## 何を比べているか

日本語の検索評価データセット JQaRA を使います。1問につき Wikipedia の文章100件が候補として与えられ、そのうちどれが「その質問に答えられる文章か」の正解ラベルが最初から付いています。評価はこの候補を並べ替える課題です。上位10件にどれだけ正解を集められたかを nDCG@10 で測ります。

正解ラベルを自分で作らずに済むのがこのデータセットを選んだ理由です。検索の比較で最も恣意的になりやすいのが「何を正解とするか」で、そこを公開データに委ねれば結果の信頼性が上がります。公開されている日本語リランカーのスコアとも同じ土俵で比べられます。

比較する経路は6つです。

| 経路 | 中身 | 位置づけ |
|---|---|---|
| `embedding` | Cohere Embed v4 で候補を類似度順に並べる | 従来のRAG。比較の基準 |
| `cohere_rerank` | Cohere Rerank v3.5 で並べ替える | 従来の再ランキング |
| `jev_pointwise` | 候補ごとにNoulを1問立て、返った確率で並べる | Jev: 関連度スコアリング |
| `jev_crossencode` | 候補1件ごとにリクエストを分けて判定する | Jev: クロスエンコード（公式レシピ） |
| `jev_pairwise` | 2候補のどちらが良いかをChoiceで比べ、勝率で並べる | Jev: ペアワイズ比較 |
| `jev_hybrid` | 埋め込みの順位とJevの判定を重み付き融合する | Jev: 埋め込みの補完 |

## 公式ユースケースとの対応

公式ドキュメントの Search and retrieval は5項目で、すべてに実装とテストを対応させています。`uv run jev-rag usecases` でも出力できます。

| 公式ユースケース | 実装 | 経路 |
|---|---|---|
| Replace or supplement embeddings in RAG pipelines | `jev_rag.rankers` | `jev_hybrid`（補完）/ `jev_crossencode`（置換） |
| Score query-to-candidate relevance | `jev_rag.jev.rerank` | `jev_pointwise` |
| Rerank results with pairwise comparisons | `jev_rag.jev.pairwise` | `jev_pairwise` |
| Cross-encode queries and candidates | `jev_rag.jev.crossencode` | `jev_crossencode` |
| Select useful context for downstream AI workflows | `jev_rag.jev.context` | `jev-rag context` |

「埋め込みの置換」と「クロスエンコード」が同じ経路なのは、候補が固定されたこの課題では両者が同じ処理になるためです。`jev_crossencode` は埋め込みモデルを一切使わないので、そのまま「置換」にあたります。無理に別の行を立てても意味のある差は出ません。

文脈選択だけは並べ替えではなく採否の判断なので、順位の指標ではなく「渡す文脈が何文字減り、正解が何割残ったか」で測ります。

## 使い方

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env

aws sso login --profile production      # Bedrock 用。鍵ではなくSSO
uv run jev-rag prepare                  # JQaRA を取得する（60MB）
uv run jev-rag check                    # Jev の疎通確認
uv run jev-rag evaluate --queries 100 --candidates 30
uv run jev-rag report --movers jev_crossencode
uv run jev-rag context                  # 文脈選択の効果を測る
```

`.env` に書く秘密情報は Vercel AI Gateway の `AI_GATEWAY_API_KEY` ひとつだけです。残りは接続先とモデルIDの指定で、秘密ではありません。TypeSafe のアカウントが開通したら `JEV_TRANSPORT=direct` と `TYPESAFE_API_KEY` に切り替えます。Bedrock の認証情報は `.env` には書きません。boto3 の既定のチェーンが `AWS_PROFILE` を見るので、プロファイル名だけ指定して `aws sso login` で認証します。`jev-rag` は起動時に `.env` を読みますが、既に export されている環境変数のほうが優先されます。

`evaluate` は結果を `out/results.json` に保存します。`report` はそこから表を出し直すので、測り直さずに見せ方だけ変えられます。`--movers` を付けると、従来手法で沈んでいた正解をどの質問で引き上げたかが一覧で出ます。数字だけでは伝わらない部分なので、記事にはこちらが効きます。

費用の目安です。Jev は入力100万トークンあたり$0.042で、出力は無料です。100問×候補30件のクロスエンコードで3,000リクエスト、おおよそ150万トークンなので$0.07程度。全経路を回しても数十円の範囲に収まります。

## Jev の2経路

同じリクエストボディを2つの宛先に送り分けるだけの設計です。`JevClient` から上のコードは経路を知りません。

アカウント開通前は Vercel AI Gateway を使います。`JEV_TRANSPORT=gateway` と `AI_GATEWAY_API_KEY` を設定すると `https://ai-gateway.vercel.sh/typesafe/v1/systemone` に `typesafe-ai/jev` として送ります。開通後は `JEV_TRANSPORT=direct` と `TYPESAFE_API_KEY` で `https://api.typesafe.ai/v1/systemone` に `jev-latest` として送ります。どちらも公式ドキュメントと照合済みです。

分割はトークン予算で行います。公式の上限は1リクエスト64kトークン（state と全質問の合計）、state と最長の質問1件で32kトークンで、質問の件数に上限はありません。日本語は1文字あたりほぼ1トークン、ASCIIは4文字あたり1トークンとして見積もっています。

## 結果を読むときの注意

日本語での性能。公式ドキュメントの Models は、Jev の主たる学習言語は英語で、CJKを含む他言語は「handled but not equally well」、非英語で使う前に自分のコンテンツで検証し confidence に注意せよと明記しています。このリポジトリは対象もクエリも判定の指示文もすべて日本語なので、英語のベンチマークで報告されている改善幅がそのまま出る前提では読めません。

候補の並び順。候補を絞る際は正解を落とさないよう残したうえで、必ずシャッフルしてから各経路に渡しています。これをしないと入力順に正解が偏り、何もしない経路でも満点が出てしまいます。実際に実装当初これが起きました。

指示文の影響。Jev の判定は `instructions` と `criteria` の書き方に左右されます。同梱の指示文は日本語で書いた初期値で、チューニングはしていません。従来手法（Cohere Rerank）は既製のサービスをそのまま呼んでいるので、この点では条件が対等ではありません。

## 構成

```
src/jev_rag/
  dataset.py   JQaRA の読み込み、サンプリング、候補のシャッフル
  metrics.py   nDCG@10 / MRR@10 / Recall@10
  rankers.py   比較する6経路
  runner.py    同じ問題を全経路に流して結果を集める
  report.py    比較表と、順位が上がった例の一覧
  bedrock.py   Cohere Embed v4 / Cohere Rerank v3.5 / GPT-5.6 Luna
  usecases.py  公式ユースケースと実装の対応（テストで固定）
  jev/         Jevクライアントと各ユースケースの実装
```

`jev/` の中身は公式ドキュメントと照合済みです。取得した公式ページは `.claude/docs/research/typesafe/` に逐語で保存してあります。

## 出典とライセンス

評価データは [JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA)（hotchpotch）です。question と answers は JAQKET 由来で CC-BY-SA-4.0、passage は Wikipedia の CC BY-SA 4.0 または GFDL です。利用にあたっては出典表示と継承が必要です。

- [TypeSafe AI ドキュメント](https://docs.typesafe.ai/) — API リファレンス、プリミティブ、ユースケース、cookbook
- [TypeSafe API with AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe) — ゲートウェイ経由の呼び出し
- [Cohere Embed v4 on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-v4.html)
- [GPT-5.6 Luna on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html)
