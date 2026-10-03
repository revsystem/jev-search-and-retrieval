"""Check the credentials before a long run, not after it.

Four measurement runs were lost to an SSO token expiring partway through.
The run is forty minutes; the check is one call.
"""

import pytest

from jev_rag.preflight import CredentialError, check_aws, check_jev


class Caller:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def get_caller_identity(self):
        self.calls += 1
        if self.error:
            raise self.error
        return {"Arn": "arn:aws:sts::123456789012:assumed-role/Dev/me"}


def test_valid_aws_credentials_pass():
    assert check_aws(Caller()) == "arn:aws:sts::123456789012:assumed-role/Dev/me"


def test_expired_aws_credentials_raise_before_the_run_starts():
    caller = Caller(error=Exception("Token has expired and refresh failed"))
    with pytest.raises(CredentialError, match="aws sso login"):
        check_aws(caller)


def test_the_underlying_error_is_kept_in_the_message():
    caller = Caller(error=Exception("Token has expired and refresh failed"))
    with pytest.raises(CredentialError, match="expired"):
        check_aws(caller)


def test_the_aws_check_costs_one_call():
    caller = Caller()
    check_aws(caller)
    assert caller.calls == 1


def test_a_working_jev_route_passes():
    from jev_rag.jev.client import JevClient
    from jev_rag.jev.transport import FakeTransport

    assert check_jev(JevClient(FakeTransport(noul=0.9))) is True


def test_a_broken_jev_route_raises_before_the_run_starts():
    class Broken:
        model = "jev-latest"

        def build_body(self, state, questions):
            return {}

        def send(self, body):
            raise RuntimeError("Jev 401 from https://api.typesafe.ai/v1/systemone")

    from jev_rag.jev.client import JevClient

    with pytest.raises(CredentialError, match="401"):
        check_jev(JevClient(Broken()))


def test_only_bedrock_routes_need_aws():
    from jev_rag.cli import _needs_aws

    assert (
        _needs_aws(["embedding"]) and _needs_aws(["jev_hybrid"]) and _needs_aws(["cohere_rerank"])
    )
    assert not _needs_aws(["jev_pointwise", "decider_single", "decider_pointwise_en"])
