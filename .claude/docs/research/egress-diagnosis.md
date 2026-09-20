# 外部サイトに到達できない原因の切り分け

実施日 2026-09-20 / 環境 `work` (`env_01NaYVSYsmQyHe1T1JaQvWgS`)

組織設定で外部サイトへのアクセスを許可しても状況が変わらなかったため、
どのホストが到達できるかを実測して原因を切り分けた。

## 実測結果

`curl -sS -o /dev/null -w "%{http_code}" --max-time 15 https://<host>/` の結果。
`000` はHTTPリクエスト送信前にプロキシのCONNECTトンネル確立が拒否されたことを示す
(`curl: (56) CONNECT tunnel failed, response 403`)。サーバ側の応答ではない。

| ホスト | コード | 分類 |
|---|---|---|
| github.com | 400 | 到達 |
| api.github.com | 200 | 到達 |
| raw.githubusercontent.com | 301 | 到達 |
| pypi.org | 200 | 到達（`noProxy` によりプロキシを迂回） |
| registry.npmjs.org | 200 | 到達（同上） |
| example.com | 000 | 遮断 |
| www.google.com | 000 | 遮断 |
| www.wikipedia.org | 000 | 遮断 |
| anthropic.com | 000 | 遮断 |
| docs.anthropic.com | 000 | 遮断 |
| docs.typesafe.ai | 000 | 遮断 |
| api.typesafe.ai | 000 | 遮断 |
| www.soumu.go.jp | 000 | 遮断 |
| vercel.com | 000 | 遮断 |
| docs.aws.amazon.com | 000 | 遮断 |

## 結論

決め手は `example.com` と `anthropic.com` も遮断されている点である。
特定のサイトが個別に拒否されているのではなく、GitHubとパッケージレジストリ以外の
すべてが遮断されている。これは「信頼できるソースのみ」の制限プリセットの挙動であり、
許可リストに項目が追加された状態ではない。

到達できるのは次の2群だけである。GitHub系はプロキシを経由したうえでポリシーに許可されている。
パッケージレジストリ (pypi.org, files.pythonhosted.org, registry.npmjs.org, jsr.io,
index.crates.io, proxy.golang.org) とAnthropicのAPIエンドポイントは `noProxy` に
列挙されており、そもそもプロキシを通らない。

## コンテナは入れ替わっている

プロキシのローカルポートはセッションをまたいで変化している
(2026-09-19 のプローブでは 44777、2026-09-20 では 33043)。
つまり設定変更後に起動した新しいコンテナでも制限プリセットのままであり、
「古いコンテナに設定が反映されていないだけ」では説明がつかない。
同一環境に新規コンテナを起動して確認した際も結果は同じだった
(`.claude/docs/research/pdf-and-docs-validation.md`)。

## 確認すべき箇所

ネットワークアクセスの設定は環境単位で、コンテナの起動時に適用される。
claude.ai/code の Environments から `work` 環境のネットワークアクセス設定を確認し、
「すべてのネットワークアクセス」またはカスタム許可リスト
(`docs.typesafe.ai`, `www.soumu.go.jp` を含む) に変更する。
設定方法は https://code.claude.com/docs/en/claude-code-on-the-web に記載がある。

変更は既存コンテナには効かないため、変更後に新しいセッションを作る必要がある。
効いているかどうかは、新セッションで次の1行を実行すれば判定できる。

```
curl -sS -o /dev/null -w "%{http_code}\n" https://example.com/
```

`200` なら開通、`000` なら制限のままである。

## 迂回について

ミラーサイト、キャッシュ、別経路のいずれも試していない。
`/root/.ccr/README.md` はポリシー拒否を迂回せず報告するよう指示しており、それに従っている。

なお、もう一方の環境 (`Default` / `env_01K42UzdQ9etxPunC4ybkK28`) でも同じ確認を試みたが、
そのセッション自身の権限分類器が複数ホストの走査を拒否したためプローブは実行されず、
当該環境の状態は未確定のままである (`.claude/docs/research/egress-probe-default-env.md`)。
