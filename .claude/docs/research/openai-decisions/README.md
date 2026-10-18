# OpenAI Decisions API の資料

2026-10-19 に取得した。Decisions API は public beta なので、仕様や料金は GA までに変わりうる。記事を直すときは取得し直して確かめる。

| ファイル | 取得元 |
|---|---|
| `guides_decisions.md` | https://developers.openai.com/api/docs/guides/decisions.md |
| `models_gpt-6-luna.md` | https://developers.openai.com/api/docs/models/gpt-6-luna.md |
| `sdk_types_decision.py.txt` | openai（Python SDK）3.35.1 の `openai/types/decision.py` |
| `sdk_types_decision_create_params.py.txt` | 同 `openai/types/decision_create_params.py` |

API リファレンス（`/api/reference/llms.txt`）には、取得時点で Decisions のページがなかった。リクエストと応答の欄は SDK の型定義で確かめた。

ガイドと SDK の記述が食い違う点が 1 つある。画像の入力について、ガイドは「base64 の data URL のみ。HTTP(S) の URL は使えない」と書き、SDK の型定義は「base64 の data URL か、公開された HTTP(S) の URL」と書いている。この検証はテキストだけを送るので影響しない。
