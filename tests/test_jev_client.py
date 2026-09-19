import pytest

from jev_rag.jev.client import MAX_QUESTIONS_PER_REQUEST, JevClient
from jev_rag.jev.questions import Noul, Score
from jev_rag.jev.transport import DirectTransport, FakeTransport, GatewayTransport, build_transport


def test_direct_transport_targets_the_typesafe_api():
    t = DirectTransport(api_key="ts-key")
    assert t.url == "https://api.typesafe.ai/v1/systemone"
    assert t.headers()["Authorization"] == "Bearer ts-key"
    assert t.model == "jev-latest"


def test_gateway_transport_targets_the_vercel_ai_gateway():
    t = GatewayTransport(api_key="vck-key")
    assert t.url == "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    assert t.headers()["Authorization"] == "Bearer vck-key"
    assert t.model == "typesafe-ai/jev"


def test_transports_share_one_request_body():
    body_direct = DirectTransport(api_key="a").build_body("state", {"q": {"type": "noul"}})
    body_gateway = GatewayTransport(api_key="b").build_body("state", {"q": {"type": "noul"}})
    assert body_direct["state"] == body_gateway["state"] == "state"
    assert body_direct["questions"] == body_gateway["questions"]
    # only the routed model id differs between the two transports
    assert body_direct["model"] != body_gateway["model"]


def test_build_transport_selects_by_name():
    assert isinstance(build_transport("direct", api_key="k"), DirectTransport)
    assert isinstance(build_transport("gateway", api_key="k"), GatewayTransport)
    assert isinstance(build_transport("fake"), FakeTransport)
    with pytest.raises(ValueError):
        build_transport("carrier-pigeon", api_key="k")


def test_transports_require_an_api_key():
    with pytest.raises(ValueError):
        DirectTransport(api_key="")


def test_client_sends_a_single_request_for_a_small_question_set():
    transport = FakeTransport(noul=0.5)
    client = JevClient(transport)
    answers = client.evaluate("state", {"a": Noul("?"), "b": Noul("?")})
    assert len(transport.requests) == 1
    assert set(answers) == {"a", "b"}


def test_client_splits_question_sets_above_the_gateway_limit():
    transport = FakeTransport(noul=0.5)
    client = JevClient(transport)
    questions = {f"q{i}": Noul("?") for i in range(MAX_QUESTIONS_PER_REQUEST + 5)}
    answers = client.evaluate("state", questions)

    assert len(transport.requests) == 2
    assert all(len(r["questions"]) <= MAX_QUESTIONS_PER_REQUEST for r in transport.requests)
    assert set(answers) == set(questions)


def test_client_keeps_the_state_identical_across_split_requests():
    transport = FakeTransport(noul=0.5)
    client = JevClient(transport)
    client.evaluate({"query": "q"}, {f"q{i}": Noul("?") for i in range(40)})
    assert {r["state"]["query"] for r in transport.requests} == {"q"}


def test_client_serialises_question_objects_to_the_wire_format():
    transport = FakeTransport(noul=0.5)
    JevClient(transport).evaluate("s", {"a": Score("?", criteria=["lo", "hi"])})
    assert transport.requests[0]["questions"]["a"] == {
        "type": "score",
        "instructions": "?",
        "criteria": ["lo", "hi"],
    }


def test_client_rejects_an_empty_question_set():
    with pytest.raises(ValueError):
        JevClient(FakeTransport()).evaluate("s", {})


def test_rate_limits_and_server_faults_are_retried():
    from jev_rag.jev.transport import should_retry

    assert should_retry(429) is True
    assert should_retry(503) is True


def test_authentication_and_schema_errors_are_not_retried():
    from jev_rag.jev.transport import should_retry

    assert should_retry(401) is False
    assert should_retry(422) is False
    assert should_retry(200) is False
