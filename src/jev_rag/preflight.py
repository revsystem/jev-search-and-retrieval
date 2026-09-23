"""Checking credentials before a long run rather than during one.

A full measurement is tens of minutes across several routes. Four runs were
lost to an SSO token expiring partway through, each time after the work had
already been paid for. These checks cost one call each.
"""

from __future__ import annotations

from typing import Any


class CredentialError(RuntimeError):
    """Raised before any measurement work is done."""


def check_aws(caller: Any) -> str:
    """Confirm the AWS credentials resolve, and return the caller's identity."""
    try:
        return caller.get_caller_identity()["Arn"]
    except Exception as error:  # noqa: BLE001 - any failure here is the same failure
        raise CredentialError(
            f"AWSの認証情報が使えない: {error}\n"
            "  `aws sso login --profile $AWS_PROFILE` を実行してから再試行する。"
        ) from error


def check_jev(client: Any) -> bool:
    """Confirm the configured Jev route answers a trivial question."""
    from jev_rag.jev.questions import Noul

    try:
        client.evaluate(
            {"query": "赤い惑星", "candidates": [{"text": "火星は赤い惑星と呼ばれる。"}]},
            {"ok": Noul("`candidates[0].text` は `query` の根拠になりますか。")},
        )
    except Exception as error:  # noqa: BLE001 - any failure here is the same failure
        raise CredentialError(
            f"Jevへのリクエストが通らない: {error}\n  .env の JEV_TRANSPORT と鍵を確認する。"
        ) from error
    return True
