# 従来RAG / rerank と Jev の比較

総務省『情報通信白書』のPDFを対象に、Amazon Bedrock と S3 Vectors で組んだRAGに対して、TypeSafe の System One モデル Jev を検索・再ランキングに挟んだ場合との差を測るための実験リポジトリです。

## 期待された成果物への回答

3点の期待のうち、2点はそのまま実装でき、1点は前提が成り立ちません。先に書いておきます。

クエリに対する関連度判定は Noul で実装しました。候補ごとに「この文書はクエリの根拠になるか」を独立した確率として返させ、閾値未満を捨てます。話題が重なるだけの断片が生成モデルに渡るのを防ぐのが目的です。`src/jev_rag/jev/rerank.py` の `relevance_filter` です。

下位に沈んだ結果の引き上げは Score で実装しました。4段階の証拠スケール（無関係 / 話題は重なるが根拠にならない / 部分的な根拠 / 直接の根拠）に候補を配置させ、小数点付きのスコアで細かく順序を付け、ベクトルスコアと重み付き融合します。同ファイルの `rerank` です。融合する設計にしたのは、Jev を単独のリランカーとして使うより、意味検索の候補の上に重ねたほうが効く、という公開実験の報告に沿わせたためです。

埋め込み精度の改善については、率直に言って Jev では実現できません。Jev は埋め込みモデルではなく、ベクトルそのものを出力しないので、Cohere Embed v4 が特定の文字列に対して返すベクトルを Jev が変えることはありません。このリポジトリで実装したのは、その代わりに成立する二つの間接的な改善です。索引時に Jev が各チャンクへ分野・記述種別・自己完結性・情報密度のラベルを付け、文脈依存の断片には節見出しを補ってから埋め込みに渡します（いわゆる contextual retrieval）。同じラベルは S3 Vectors のフィルタ可能メタデータになり、クエリ側でも Jev が分野を判定して、絞り込みが安全なときだけメタデータフィルタを掛けます。埋め込みモデルの精度ではなく、埋め込む文字列と検索範囲を変えることで検索精度を上げるアプローチです。`src/jev_rag/jev/enrich.py` にあります。

## 構成

取り込みは PDF をページ単位で読み、日本語向けに正規化してから重なり付きのウィンドウに分割します。PDFの行折り返しは日本語では余計な空白になるため、前後がともに日本語文字なら空白を入れずに連結し、全角スペースと連続空行を畳んでいます。埋め込みは Bedrock 経由の Cohere Embed v4、ベクトルストアは S3 Vectors、回答生成は Bedrock Converse API 経由の GPT-5.6 Luna です。従来型リランカーの比較対象として Bedrock Rerank API の Cohere Rerank v3.5 を使います。

比較するパイプラインは4本です。`baseline` はベクトル検索のみ、`classic_rerank` は候補30件を Cohere Rerank で並べ替え、`jev_rerank` は候補30件を Jev の Noul で絞り Score で並べ替え、`jev_full` はさらに索引時エンリッチとクエリ計画によるメタデータ絞り込みを加えます。

```
src/jev_rag/
  documents.py     PDF読み込み、日本語正規化、チャンク分割
  bedrock.py       Cohere Embed v4 / Cohere Rerank v3.5 / GPT-5.6 Luna
  vector_store.py  S3 Vectors とオフライン用のインメモリ実装
  pipelines.py     比較対象4本
  evaluation.py    nDCG / Recall / Precision / MRR とキーワードルールによる正解付け
  demo.py          認証情報なしで動くオフラインデモ
  jev/
    questions.py   Noul / Choice / Score の型とレスポンス解析
    transport.py   AI Gateway 経路、TypeSafe 直接経路、オフライン用の偽経路
    client.py      32問ごとの分割と並列送信
    rerank.py      関連度フィルタとスコアリング
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

認証情報なしで比較ハーネスの動作だけ確認するなら次のコマンドです。

```bash
uv run jev-rag demo
```

実データで測る場合は取り込みから行います。`--enrich` を付けると索引時に Jev を通します。

```bash
uv run jev-rag ingest --enrich
uv run jev-rag labels                   # 正解ルールのマッチ件数を確認する
uv run jev-rag evaluate --top-k 5
uv run jev-rag ask "生成AIの利用率は前年と比べてどう変化したか"
```

## 評価設計

`data/eval/queries.yaml` のクエリには、chunk_id ではなくキーワードルールで正解を記述しています。チャンク分割の設定を変えると chunk_id が変わってしまい、直接書くと再現できないためです。等級は3が直接の根拠、2が部分的な根拠、1が話題は重なるが根拠にならない断片を表します。

等級1を「関連あり」に数えないのは意図的です。等級1はまさにこの比較が問題にしている「語彙は重なるが答えではない断片」なので、これをヒットとして数えると、改善したいはずの失敗を評価指標が褒めてしまいます。nDCG は全等級の利得を使い、Recall / Precision / Hit / MRR は等級2以上を関連ありとして計算します。

同梱のキーワードルールは白書の一般的な語彙を前提に書いた初期値で、実際のPDFに対して検証できていません。取り込み後に `jev-rag labels` でマッチ件数を確認し、0件や過剰マッチのルールは調整してください。

## オフラインデモの結果

`uv run jev-rag demo` は合成コーパス12件とクエリ4件で全パイプラインを動かします。

```
| pipeline | ndcg@5 | strict_ndcg@5 | recall@5 | precision@5 | mrr |
|---|---|---|---|---|---|
| baseline | 0.862 | 0.815 | 1.000 | 0.300 | 0.750 |
| baseline_enriched | 0.862 | 0.815 | 1.000 | 0.300 | 0.750 |
| jev_rerank | 0.901 (+0.039) | 1.000 (+0.185) | 1.000 | 0.300 | 1.000 (+0.250) |
| jev_full | 0.855 (-0.007) | 1.000 (+0.185) | 0.875 (-0.125) | 0.250 (-0.050) | 1.000 (+0.250) |
```

ここで Jev が返す値は実際のモデル出力ではなく、`data/demo/offline.yaml` に手で書いた台本です。したがってこの表が示しているのは比較ハーネスの配線が正しいことだけで、Jev の実力の証拠ではありません。循環を隠さないためにフィクスチャは外出しにしてあります。

それでも読み取れることが3つあります。ベースラインは「本節では生成AIの動向について整理する。生成AIの利用率や生成AIの活用状況を…」のような、クエリ語を繰り返すだけで答えを含まない断片を1位に置き、実際の数値を含む断片を2位に沈めます。狙った失敗形が再現できています。Jev のスコアリングはそれを引き上げ、MRR が 0.750 から 1.000 に、直接の根拠だけを見る strict_nDCG が 0.815 から 1.000 になります。そして `jev_full` のメタデータ絞り込みは nDCG をわずかに下げます。分野フィルタが、別分野に分類された部分的な根拠（諸外国比較のチャンク）を候補から外してしまうためで、絞り込みには再現率を削るリスクがあることをハーネスが検出できています。

オフラインの埋め込みは文字バイグラムの語彙一致に置き換えてあるため、索引時エンリッチの効果は `baseline_enriched` に現れていません。ヘッダを足しても同一分野の全チャンクに同じバイグラムが乗るだけで順序が変わらないからです。エンリッチの効果を測るには実際の埋め込みモデルが要ります。

## 検証できていないこと

この環境からは `docs.typesafe.ai`、`vercel.com`、`soumu.go.jp` のいずれにも到達できず、Jev の API キーも AWS の認証情報もありません。したがって次は未検証です。Jev への実リクエストは一度も送っていません。対象PDFの取り込みも行っていないため、チャンク分割の妥当性と評価クエリのキーワードルールは実データで確認できていません。Bedrock と S3 Vectors の呼び出しも実行していません。

Jev のリクエスト・レスポンス形式は公式ドキュメントに到達できなかったため、下記の二次資料から組み立てています。特に次の2点は実際に叩いて確認してください。AI Gateway の TypeSafe 互換エンドポイントで受け付けられる model id が `typesafe-ai/jev` か `jev-latest` か（`JEV_GATEWAY_MODEL` で切り替えられるようにしてあります）。もう1点は S3 Vectors のインデックス作成 API の boto3 メソッド名で、資料によって `create_index` と `create_vector_index` が混在していたため、両方を試す実装にしています。

テストは101件が通り、ruff も通ります。ただしこれらはすべて、ネットワークに出ない範囲の純粋なロジックと配線に対するテストです。

## 出典

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
