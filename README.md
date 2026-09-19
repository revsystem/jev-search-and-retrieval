# 従来RAG / rerank と Jev の比較

総務省『情報通信白書』のPDFを対象に、Amazon Bedrock と S3 Vectors で組んだRAGに対して、TypeSafe の System One モデル Jev を検索・再ランキングに挟んだ場合との差を測るための実験リポジトリです。実装は公式ドキュメントの Search and retrieval ユースケース5項目に1対1で対応させています。

## 公式ユースケースとの対応

公式ドキュメントの Search and retrieval は次の5項目で、すべてに実装とテストを対応させています。決定の型は同ページの Example task categories 表の区分です。

| 公式ユースケース | 決定の型 | 実装 | 実行方法 | テスト |
|---|---|---|---|---|
| Replace or supplement embeddings in RAG pipelines with semantic search, scoring, and ranking. | Search, Retrieval, Ranking | `jev_rag.jev.retrieval` | `jev_only, jev_hybrid` | `tests/test_uc1_jev_retrieval.py` |
| Score query-to-candidate relevance. | Scoring, Ranking | `jev_rag.jev.rerank` | `jev_noul` | `tests/test_uc2_relevance_scoring.py` |
| Rerank results with pairwise comparisons. | Ranking | `jev_rag.jev.pairwise` | `jev_pairwise` | `tests/test_uc3_pairwise.py` |
| Cross-encode queries and candidates for higher precision. | Scoring, Ranking | `jev_rag.jev.crossencode` | `jev_crossencode` | `tests/test_uc4_cross_encode.py` |
| Select useful context for downstream AI workflows. | Retrieval | `jev_rag.jev.context` | `jev-rag ask --context-budget` | `tests/test_uc5_context_selection.py` |

この対応表は README だけでなく `src/jev_rag/usecases.py` にレジストリとして持たせてあり、`uv run jev-rag usecases` で出力できます。モジュール名やパイプライン名を変えたりテストを消したりすると `tests/test_usecases_registry.py` が落ちるので、表と実装がずれたまま放置されることはありません。

それぞれの設計の違いは次のとおりです。`jev_only` は埋め込みモデルを経路から完全に外し、コーパスを32問ずつのNoulで走査して確率で順位を付けます。インデックス規模ではなくコストで頭打ちになるので、1文書やメタデータで絞った範囲のように既に狭まったコーパス向けです。`jev_hybrid` は埋め込みを再現率の段、Jevを適合率の段として融合します。`jev_noul` は候補一覧を共有stateに置いて候補ごとにNoulを1問ずつ立てます。`jev_pairwise` は絶対尺度を避け、2候補のどちらが良いかというChoiceだけで順位を決めます。比較数が増える代わりに、リクエスト内の質問は並列評価されるため1ラウンドの実時間は1問とほぼ変わりません。`jev_crossencode` はペアごとにリクエストを分け、stateに他の候補を一切入れないことで隣接候補による汚染を避けます。文脈選択は順位ではなく採否を決める段で、有用性と重複を判定してから予算内に貪欲に詰めます。

比較対象として、ベクトル検索のみの `baseline` と、Bedrock Rerank API の Cohere Rerank v3.5 を使う `classic_rerank` を置いています。これに加えて、索引時エンリッチとクエリ計画を重ねた `jev_full` があります。この2つは公式マップの項目ではなく、Jevを挟んだ効果を測るための対照群と拡張です。

なお同ページには Universal Verification（引用誤り・ハルシネーションの検出）や LLM guardrails（検索した文章に混入したプロンプトインジェクションの検出）といった隣接カテゴリもあり、RAGにそのまま足せる内容ですが、今回の依頼範囲である Search and retrieval の外なので実装していません。

## 期待された成果物への回答

クエリに対する関連度判定は Noul で実装しました。候補ごとに「この文書はクエリの根拠になるか」を独立した確率として返させ、閾値未満を捨てます。話題が重なるだけの断片が生成モデルに渡るのを防ぐのが目的です。`jev/rerank.py` の `relevance_filter` と `noul_rerank` です。

下位に沈んだ結果の引き上げは Score と Choice の両方で実装しました。`rerank` は4段階の証拠スケール（無関係 / 話題は重なるが根拠にならない / 部分的な根拠 / 直接の根拠）に候補を配置させ、小数点付きのスコアをベクトルスコアと重み付き融合します。`PairwiseReranker` は絶対尺度を使わず、勝率で並べ替えます。

埋め込みについては、公式ドキュメントが挙げているのは replace or supplement、つまり埋め込みによる検索段そのものをJevで置き換えるか補うかであって、埋め込みベクトルの生成ではありません。Jev はベクトルを出力しないので、Cohere Embed v4 が特定の文字列に対して返すベクトルをJevが変えることはできません。このリポジトリではその区別に沿って3通りを用意しています。`jev_only` は埋め込みを使わずJevだけで検索します。`jev_hybrid` は埋め込みの候補にJevの判定を重ねます。そして索引時エンリッチ（`jev/enrich.py`）では、Jevが各チャンクへ分野・記述種別・自己完結性・情報密度のラベルを付け、文脈依存の断片には節見出しを補ってから埋め込みに渡します。同じラベルは S3 Vectors のフィルタ可能メタデータになり、クエリ側でもJevが分野を判定して絞り込みが安全なときだけフィルタを掛けます。埋め込みモデルの精度ではなく、埋め込む文字列と検索範囲を変えることで検索精度を上げるアプローチです。

## 構成

取り込みは PDF をページ単位で読み、日本語向けに正規化してから重なり付きのウィンドウに分割します。PDFの行折り返しは日本語では余計な空白になるため、前後がともに日本語文字なら空白を入れずに連結し、全角スペースと連続空行を畳んでいます。埋め込みは Bedrock 経由の Cohere Embed v4、ベクトルストアは S3 Vectors、回答生成は Bedrock Converse API 経由の GPT-5.6 Luna です。

```
src/jev_rag/
  documents.py     PDF読み込み、日本語正規化、チャンク分割
  bedrock.py       Cohere Embed v4 / Cohere Rerank v3.5 / GPT-5.6 Luna
  vector_store.py  S3 Vectors とオフライン用のインメモリ実装
  pipelines.py     比較対象パイプライン
  evaluation.py    nDCG / Recall / Precision / MRR とキーワードルールによる正解付け
  demo.py          認証情報なしで動くオフラインデモ
  jev/
    questions.py   Noul / Choice / Score の型とレスポンス解析
    transport.py   AI Gateway 経路、TypeSafe 直接経路、オフライン用の偽経路
    client.py      32問ごとの分割と並列送信
    prompts.py     判定の指示文と4段階スケール
    retrieval.py   埋め込みの置き換えと併用
    rerank.py      関連度スコアリングとスコア融合
    pairwise.py    ペアワイズ比較
    crossencode.py クエリと候補のクロスエンコード
    context.py     下流ワークフローへ渡す文脈の選別
    enrich.py      索引時エンリッチとクエリ計画
```

## Jev の2経路

同じリクエストボディを2つの宛先に送り分けるだけの設計にしてあります。違うのはホストと鍵と model id だけで、`JevClient` から上のコードは経路を知りません。

アカウント開通前は Vercel AI Gateway を使います。`JEV_TRANSPORT=gateway` とし、`AI_GATEWAY_API_KEY` を設定すると `https://ai-gateway.vercel.sh/typesafe/v1/systemone` に `typesafe-ai/jev` として送ります。開通後は `JEV_TRANSPORT=direct` に切り替え、`TYPESAFE_API_KEY` を設定すると `https://api.typesafe.ai/v1/systemone` に `jev-latest` として送ります。疎通確認は `jev-rag check --transport gateway` です。

リクエストは共有された `state` に対して名前付きの質問を並べる形です。質問は並列かつ独立に評価されるため、候補30件の関連度判定を1リクエストにまとめても、1件のときとレイテンシがほとんど変わりません。ゲートウェイの上限である1リクエスト32問を超える場合は `JevClient` が同一 state のまま分割します。

## セットアップ

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # 鍵とモデルIDを設定する
```

認証情報なしで全ユースケースの動作を確認するなら次のコマンドです。

```bash
uv run jev-rag demo
```

実データで測る場合は取り込みから行います。`--enrich` を付けると索引時に Jev を通します。

```bash
uv run jev-rag ingest --enrich
uv run jev-rag labels                   # 正解ルールのマッチ件数を確認する
uv run jev-rag evaluate --top-k 5 \
  --pipelines baseline,classic_rerank,jev_only,jev_hybrid,jev_noul,jev_pairwise,jev_crossencode,jev_full
uv run jev-rag ask "生成AIの利用率は前年と比べてどう変化したか" \
  --pipeline jev_crossencode --context-budget 4000
```

## 評価設計

`data/eval/queries.yaml` のクエリには、chunk_id ではなくルールで正解を記述しています。チャンク分割の設定を変えると chunk_id が変わってしまい、直接書くと再現できないためです。等級は3が直接の根拠（具体的な数値を伴う）、2が部分的な根拠（主題と論点は合うが数値がない）、1が話題は重なるが根拠にならない断片です。ルールは `all` / `any` / `none` のキーワード条件と、それとANDで効く `regex` を取ります。

等級1を「関連あり」に数えないのは意図的です。等級1はまさにこの比較が問題にしている「語彙は重なるが答えではない断片」なので、これをヒットとして数えると、改善したいはずの失敗を評価指標が褒めてしまいます。nDCG は全等級の利得を使い、Recall / Precision / Hit / MRR は等級2以上を関連ありとして計算します。

等級3の条件に `regex` で数値を要求しているのは、統計白書では `%` という記号だけならほぼ全ページに現れるためです。主題語を含むチャンクの大半が「直接の根拠」に昇格してしまうと、strict_ndcg の母集団が膨らんで指標が何も区別しなくなります。数値を伴う実績値だけを拾うようにしてあります。

取り込み時に本文を NFKC 正規化しているのも同じ理由からです。PDFのテキスト抽出は全角英数字や半角カタカナを普通に返すので、正規化しないと `生成ＡＩ` と書かれた文書に対して `生成AI` を要求するルールが一件もマッチせず、しかも0件であることに気づきにくいという失敗をします。節見出しは正規化前の行から取るため元の文字のまま残ります。

`jev-rag labels` はルールの較正を両方向から確認します。0件マッチのクエリだけでなく、等級3がコーパスの一定割合（既定10%、`--max-share` で変更可）を超えたクエリも過剰マッチとして警告します。等級2のラベルがどのクエリにも生成されていない場合は、Recall / Precision / MRR が実質的に等級3のみで決まっている旨を最後に表示します。

同梱のルールは対象PDF（令和8年版 第1章「個人におけるAI利用」、15ページ）を実際に取り込み、37チャンクの語彙を数えたうえで較正してあります。別の文書を対象にする場合は `jev-rag labels` を実行し、両方向の警告を見て調整してください。認証情報がなくても `jev-rag evaluate --offline-embeddings` で字句ベースの代用埋め込みによる baseline を測れるので、ラベルが退化していないかはそこで確認できます（実コーパスでの baseline は nDCG@5 = 0.322）。

対象PDFの取り込みで分かったことは `.claude/docs/research/pdf-and-docs-validation.md` にまとめてあります。抽出結果には装飾用の字間（`個 人 に お け る A I 利 用`）、太字表現の二重描画（`26.726.7`）、全ページに繰り返される柱と図表軸が含まれ、正規化前は50チャンク中13チャンクがその重複で、節見出しは3チャンクにしか付いていませんでした。特に重要だったのは、本文が `生成 AI サービス` と組版空白付きで抽出されるため、`生成AI` を要求するルールが該当箇所を取りこぼしていた点です。エラーにならず0件になるだけなので、実データを通さないと発見できませんでした。正規化を直した結果、チャンクは37、節見出しの付与は37/37になっています。

## オフラインデモの結果

`uv run jev-rag demo` は合成コーパス13件とクエリ4件で全ユースケースを動かします。

```
| pipeline | ndcg@5 | strict_ndcg@5 | recall@5 | precision@5 | mrr |
|---|---|---|---|---|---|
| baseline | 0.867 | 0.831 | 1.000 | 0.350 | 0.750 |
| jev_only | 0.972 (+0.105) | 1.000 (+0.169) | 1.000 | 0.350 | 1.000 (+0.250) |
| jev_hybrid | 0.999 (+0.131) | 1.000 (+0.169) | 1.000 | 0.350 | 1.000 (+0.250) |
| jev_noul | 0.972 (+0.105) | 1.000 (+0.169) | 1.000 | 0.350 | 1.000 (+0.250) |
| jev_pairwise | 0.948 (+0.080) | 1.000 (+0.169) | 1.000 | 0.350 | 1.000 (+0.250) |
| jev_crossencode | 0.972 (+0.105) | 1.000 (+0.169) | 1.000 | 0.350 | 1.000 (+0.250) |
| jev_full | 0.892 (+0.024) | 1.000 (+0.169) | 0.917 (-0.083) | 0.300 (-0.050) | 1.000 (+0.250) |

文脈選択（予算 200 文字、入力は jev_noul の上位6件）:
| query | 候補 | 採用 | 文字数 | 除外理由 |
|---|---|---|---|---|
| d1 | 5 | 2 | 267 -> 118 | c05:not_useful, c06:not_useful, c13:redundant |
| d2 | 5 | 2 | 272 -> 119 | c01:not_useful, c02:not_useful, c05:not_useful |
| d3 | 5 | 1 | 243 -> 60 | c03:not_useful, c05:not_useful, c11:not_useful, c12:not_useful |
| d4 | 5 | 2 | 263 -> 106 | c03:not_useful, c05:not_useful, c06:not_useful |
```

ここで Jev が返す値は実際のモデル出力ではなく、`data/demo/offline.yaml` に手で書いた台本です。したがってこの表が示しているのは比較ハーネスの配線が正しいことだけで、Jev の実力の証拠ではありません。循環を隠さないためにフィクスチャは外出しにしてあります。台本は候補ごとの関連度を1か所に書き、ペアワイズの勝敗はそこから確率として導出しているので、5つの経路に別々の都合の良い数字を置くことはできない構造にしています。

それでも読み取れることがいくつかあります。ベースラインは「本節では生成AIの動向について整理する。生成AIの利用率や生成AIの活用状況を…」のような、クエリ語を繰り返すだけで答えを含まない断片を1位に置き、実際の数値を含む断片を沈めます。狙った失敗形が再現できています。Jevを挟んだ5経路はいずれもこれを引き上げ、直接の根拠だけを見る strict_nDCG が 0.831 から 1.000 に、MRR が 0.750 から 1.000 になります。

経路ごとの差も出ています。`jev_hybrid` が最も高いのは、ベクトル順位とJev判定が互いの誤りを打ち消すためで、Jev単独をリランカーとして使うより意味検索の候補に重ねたほうが効くという公開実験の報告と方向が一致します。`jev_pairwise` がやや低いのは、候補13件では総当たりが1リクエストの32問上限を超え、各候補3対戦のサンプリングに落ちるためで、対戦数と精度のトレードオフがそのまま出ています。`jev_full` のメタデータ絞り込みは再現率を下げます。分野フィルタが、別分野に分類された部分的な根拠（諸外国比較のチャンク）を候補から外してしまうためで、絞り込みには再現率を削るリスクがあることをハーネスが検出できています。

文脈選択は、d1で近重複チャンク c13 を redundant として落としています。同じ統計を2回渡すのは予算の無駄であるだけでなく、生成モデルに誤った裏付け感を与えるため、順位付けとは別に落とす価値があります。

## 検証できていないこと

対象PDFは会話への添付で受領し、取り込みまで検証済みです。一方この環境からは `docs.typesafe.ai`、`vercel.com`、`soumu.go.jp`、`docs.aws.amazon.com` のいずれにも到達できません（egressゲートウェイがCONNECTに403を返す `connect_rejected` で、迂回はしていません）。同一環境に新しいコンテナを立てて再確認しても結果は同じでした。その記録は `.claude/docs/research/pdf-and-docs-validation.md` にあります。Jev の API キーも AWS の認証情報もありません。したがって次は未検証です。Jev への実リクエストは一度も送っていません。Bedrock と S3 Vectors の呼び出しも実行していないため、実際の埋め込み生成、ベクトル検索、Jevを挟んだ5経路の比較は動かせていません。チャンク分割と評価ラベルは実PDFで検証済みです。

ユースケースの一覧と決定の型は、利用者から提供された公式ドキュメント本文に基づいています。一方でリクエスト・レスポンス形式は公式ドキュメントに到達できなかったため、下記の二次資料から組み立てています。特に次の2点は実際に叩いて確認してください。AI Gateway の TypeSafe 互換エンドポイントで受け付けられる model id が `typesafe-ai/jev` か `jev-latest` か（`JEV_GATEWAY_MODEL` で切り替えられるようにしてあります）。もう1点は S3 Vectors のインデックス作成 API の boto3 メソッド名で、資料によって `create_index` と `create_vector_index` が混在していたため、両方を試す実装にしています。公式ドキュメントの目次は `https://docs.typesafe.ai/llms.txt` にありますが、これも当環境からは到達できません。

テストは221件が通り、ruff も通ります。ただしこれらはすべて、ネットワークに出ない範囲の純粋なロジックと配線に対するテストです。

## 出典

- [Example use cases — TypeSafe AI](https://docs.typesafe.ai/concepts/use-case-map#search-and-retrieval) — Search and retrieval の5項目（本文は利用者提供、当環境からは未到達）
- [Jev exploration: API schema and examples](https://github.com/SamuelSacco/jev-exploration) — `/v1/systemone` のリクエスト・レスポンス形状、価格、入力上限
- [Vessel issue #113: TypeSafe System One format adapter](https://github.com/spenceclark/Vessel/issues/113) — Noul / Choice / Score の criteria と answers スキーマ
- [TypeSafe Jev project reference](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965) — 3プリミティブの設計思想、選択肢255件の上限
- [hotchpotch/jev-reranker](https://github.com/hotchpotch/jev-reranker) — 関連度リランキングの指示文設計、閾値0.2、listwise / pointwise の分割方針
- [yibie/awesome-jev](https://github.com/yibie/awesome-jev) — 検索・検索結果再ランキングの事例とゲートウェイ経路
- [TypeSafe API with AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe) — `https://ai-gateway.vercel.sh/typesafe` 経由の呼び出し
- [Cohere Embed v4 on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-v4.html) — モデルID、出力次元
- [GPT-5.6 Luna on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html) — クロスリージョン推論プロファイル `us.openai.gpt-5.6-luna`
- [Tutorial: Getting started with S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html) — put_vectors / query_vectors
- [PDFを使ったRAGの実装](https://qiita.com/revsystem/items/dd9688ca06d952efecf1) — 参考にした記事
