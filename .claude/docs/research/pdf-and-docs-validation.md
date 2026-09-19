# PDF取り込みとドキュメント取得の検証記録

実施日 2026-09-19 / ブランチ `claude/jev-rag-comparison-s65vsm`

結論を先に書くと、指示された外部ホストは2つとも実行環境のネットワークポリシーで遮断されており、
PDFの実データを使った取り込み検証（STEP 2）とTypeSafeドキュメントの保存（STEP 3）はどちらも実施できていない。
このファイルに記録できたのは、疎通確認の正確な結果と、コードを読んで静的に確認できた範囲のラベル設計上の懸念だけである。
チャンク統計とコーパス語彙にもとづくルール提案は、実データが無いため未実施のまま残っている。

## 疎通確認の結果

| URL | HTTPステータス | curl終了コード | 詳細 |
|---|---|---|---|
| https://docs.typesafe.ai/llms.txt | 000（応答本体に到達せず） | 56 | `curl: (56) CONNECT tunnel failed, response 403` |
| https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r08/pdf/n1110000.pdf | 000（応答本体に到達せず） | 56 | `curl: (56) CONNECT tunnel failed, response 403` |

`%{http_code}` が `000` なのは、HTTPリクエストが送信される前にプロキシのCONNECTトンネル確立が拒否されているためで、
サーバ側が403を返したという意味ではない。403を返しているのはエージェントプロキシのゲートウェイである。

`curl -sS "$HTTPS_PROXY/__agentproxy/status"` が返した該当部分を原文のまま引用する。

```json
"recentRelayFailures": [
  {
    "ts": "2026-09-19T22:51:43.028Z",
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "docs.typesafe.ai:443"
  },
  {
    "ts": "2026-09-19T22:51:44.238Z",
    "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "www.soumu.go.jp:443"
  }
]
```

拒否はcurl固有の問題ではなくゲートウェイ層で起きているため、別のHTTPクライアントに変えても結果は変わらない。
指示どおり迂回は試していない（ミラーサイト、キャッシュ、別経路のいずれも試行していない）。

なお、プロキシの `noProxy` に含まれるホストは到達できる。pypi.org と files.pythonhosted.org がこれに該当するため、
依存関係のインストールだけは成功している。

## STEP 2 の実施状況

PDFが取得できないため、指示されたコマンド列は途中で停止した。実行結果を順に記録する。

PDF取得は失敗した。`data/raw/` は作成されたが空のままである。

```
$ mkdir -p data/raw && curl -sS -o data/raw/n1110000.pdf "https://www.soumu.go.jp/.../n1110000.pdf"
curl: (56) CONNECT tunnel failed, response 403
curl exit: 56
$ ls -la data/raw/
total 8
drwxr-xr-x 2 root root 4096 Sep 19 22:52 .
drwxr-xr-x 5 root root 4096 Sep 19 22:52 ..
```

仮想環境の構築と依存関係のインストールは成功した。Python 3.11.15、uv 0.8.17。
pypdf 6.19.0、boto3、pyyaml、httpx、pytest 9.1.1、ruff 0.16.8 が入っている。

取り込みコマンドは入力ファイルが無いため失敗した。

```
$ .venv/bin/jev-rag ingest --pdf data/raw/n1110000.pdf --dry-run
  File "/home/user/jev-search-and-retrieval/src/jev_rag/documents.py", line 94, in load_pdf
    reader = PdfReader(str(path))
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/n1110000.pdf'
exit: 1
```

ラベル確認コマンドも、取り込みが走っていない以上キャッシュが無く、想定どおりのガードで停止した。

```
$ .venv/bin/jev-rag labels
data/cache/chunks.jsonl がありません。先に `jev-rag ingest` を実行してください。
exit: 1
```

したがって以下はすべて未取得である。総チャンク数、ページ数、チャンク長の中央値・最小・最大、
`section` が空でないチャンクの件数、代表チャンクの本文、`jev-rag labels` の各行。
これらを推測で埋めることはしていない。

参考として、ネットワークに依存しない範囲の健全性だけは確認した。テストは181件すべて成功している。

```
$ .venv/bin/python -m pytest -q
181 passed in 0.40s
```

ただしこれはユニットテストが通ることを示すだけで、実PDFに対する日本語正規化とチャンク分割の妥当性を
裏づけるものではない。判断に必要な証拠は依然として欠けている。

## ラベル較正について（実データ無しでの静的レビュー）

指示は「実際に観測したチャンクの語彙にもとづいて置換ルールを提案する」ことだった。
コーパスが手元に無いため、その形での提案はできない。observed vocabulary が一語も無い状態で
置換語を並べても、それは提案ではなく当て推量になる。

代わりに、`src/jev_rag/evaluation.py` と `data/eval/queries.yaml` を読んで判断できる範囲の
構造的な懸念だけを記録する。以下はコードとYAMLから導いた事実であり、コーパスで検証された所見ではない。
PDFが取得できるようになった時点で、最初に確認すべき項目という位置づけである。

判定ロジックの要点は `_matches`（evaluation.py:95）にある。`all` は全語がテキストに含まれること、
`any` は1語以上含まれること、いずれも単純な部分文字列一致で、形態素解析も正規化も挟まない。
照合対象は `chunk.text` であり `embedding_text()` ではないため、`section` の見出し文字列は判定に寄与しない。

第一の懸念は、パーセント記号による過剰マッチである。q01 は `all: ["生成AI"]` と
`any: ["利用率", "利用状況", "％", "%"]`、q03 は `all: ["インターネット"]` と `any: ["利用率", "％", "%"]` を
grade 3 の条件にしている。統計白書では数値表を含むほぼ全ページにパーセント記号が現れるため、
主語となる語を含むチャンクのほとんどが「直接の根拠」に昇格しうる。
grade 3 は strict_ndcg の母数を決めるので、ここが緩いと strict 系の指標がほぼ意味を失う。

第二に、半角全角の揺れを吸収する処理がどこにも無い。`normalise_japanese_text`（documents.py:35）が
行っているのは空白の圧縮、全角空白の半角化、ハードラップされた行の結合、空行の畳み込みだけで、
`unicodedata.normalize` の呼び出しはリポジトリ全体に存在しない（`grep -rn "unicodedata\|NFKC" src/ tests/` は0件）。
`％` と `%` を両方列挙しているのは正しい対処だが、同じ問題は他の語にもある。
PDF抽出結果が `生成ＡＩ` や `ＡＩ` のような全角英字を含む場合、`生成AI` を要求する q01 / q02 / q08 は
一切マッチしなくなる。この3クエリが揃って0件になったら、まず疑うべきはここである。

第三に、grade 2 を生成するルールが1つも存在しない。8クエリすべてが grade 3 と grade 1 のみを付与する。
一方で `EVIDENCE_GRADE = 2`（evaluation.py:69）により recall / precision / mrr は grade 2 以上でしきい値を切るので、
これらの指標は grade 3 集合だけで決まる。ndcg と strict_ndcg の差は「話題が重なるだけのチャンクを
理想順位に含めるかどうか」の差でしかない。grade 1 の `partial` を広く取るほど ndcg の理想DCGが膨らみ、
どのパイプラインもスコアが下がる方向に効く。q01 / q02 / q08 の `partial` は `all: ["生成AI"]` だけなので、
生成AIに言及する全チャンクが grade 1 になる。白書の該当節では相当な割合を占める可能性が高い。

第四に、`jev-rag labels` の警告条件（cli.py:205）は `max(qrels.values(), default=0) < 2` である。
grade 3 のルールが0件で grade 1 の `partial` だけがマッチした場合も「要調整」として正しく拾われる。
逆に、grade 3 が過剰にマッチしている場合は警告が出ない。過剰側は件数を目視で確認する必要がある。

第五に、表記揺れの候補がいくつかある。q06 の `トラヒック` は総務省系の文書で使われる表記だが、
`トラフィック` と併記される文書もある。q05 の `偽・誤情報` は中黒を含む複合語で、
PDF抽出で中黒の文字コードが異なると一致しない。これらは実データを見れば一目で分かる種類の問題である。

第六に、チャンク境界の影響がある。`chunk_size=700` / `chunk_overlap=120`（config.py:93-94）で、
オーバーラップが120文字あるため通常の語がまたぎで失われることは考えにくい。
行の折り返しについては `chunk_pages` が正規化後のテキストを窓分割するので、
ハードラップ由来の分断は判定前に解消されている。この点は設計上問題ない。

PDFが取得できた時点で実施すべき確認手順は次のとおりである。
まず `jev-rag ingest --pdf ... --dry-run` の後に `data/cache/chunks.jsonl` から語ごとの出現チャンク数を数え、
`生成AI` `生成ＡＩ` `インターネット` `％` `%` `トラヒック` `トラフィック` `偽・誤情報` `情報通信産業` の
実際の頻度を出す。次に `jev-rag labels` の各行を見て、0件のクエリと、コーパスの1割を超えて
grade 3 が付いているクエリを両側から洗い出す。そのうえで、観測された語彙にもとづいて
`％` / `%` のような汎用記号を、実際に数値が記載されている文脈に固有の語（節見出しの語や図表見出しの語）へ
置き換える。現時点でその置換語を書き下すことはできない。

## STEP 3 保存したドキュメント

なし。docs.typesafe.ai へのCONNECTがゲートウェイで403となり、
`llms.txt` を含めどのページも取得できていないため、`.claude/docs/research/typesafe/` は作成していない。
再ランキングやRAGパッセージ分類に関するcookbookページについても、
`llms.txt` が読めない以上どのページが該当するか特定できていない。

## 再実行に必要な条件

この検証を完了させるには、実行環境のネットワークポリシーで `www.soumu.go.jp` と `docs.typesafe.ai` への
アウトバウンドHTTPSを許可する必要がある。環境設定の変更方法は
https://code.claude.com/docs/en/claude-code-on-the-web に記載がある。
あるいは、PDFをリポジトリ外から手元に配置してパスを渡せば、STEP 2 だけは先行して実施できる。
