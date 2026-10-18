"""Per-route usage: the model that answered, the tokens billed, and answers that never came."""

from concurrent.futures import ThreadPoolExecutor

from jev_rag.usage import UsageMeter


def test_the_meter_adds_up_every_request():
    meter = UsageMeter()
    meter.record(model="m-1", input_tokens=100, seconds=0.5, refusals=1)
    meter.record(model="m-1", input_tokens=50, seconds=0.25, missing=2)
    assert meter.snapshot() == {
        "requests": 2,
        "input_tokens": 150,
        "refusals": 1,
        "missing": 2,
        "models": ["m-1"],
        "request_seconds": [0.5, 0.25],
    }


def test_reset_starts_a_new_route_from_zero():
    meter = UsageMeter()
    meter.record(model="m-1", input_tokens=100, seconds=0.5)
    meter.reset()
    assert meter.snapshot()["requests"] == 0
    assert meter.snapshot()["models"] == []


def test_requests_from_worker_threads_are_all_counted():
    meter = UsageMeter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.record(model="m", input_tokens=1, seconds=0.0), range(400)))
    assert meter.snapshot()["input_tokens"] == 400
