# テストの流し方

テストは `uv run python -m pytest -q` で流す。`uv run pytest` は使わない。

既存のテストの一部（`tests/test_cli_judge.py` など）は `from tests.test_synthetic import ...` のように `tests` パッケージから import している。`python -m pytest` は作業ディレクトリを `sys.path` に入れるので通るが、`uv run pytest` は入れないので、収集の段階で `ModuleNotFoundError: No module named 'tests'` になる（2026-10-19、Decisions API の比較の実装中に遭遇）。
