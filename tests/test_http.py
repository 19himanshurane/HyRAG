import httpx
import pytest

from hyrag import http
from hyrag.http import DeadlineExceeded, deadline, post_json, remaining


def client(handler, timeout=30.0):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)


@pytest.fixture
def slept(monkeypatch):
    waited = []
    monkeypatch.setattr(http.time, "sleep", waited.append)
    return waited


def test_no_budget_means_no_deadline():
    assert remaining() is None


def test_each_attempt_timeout_is_cut_to_the_time_left():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"ok": True})

    with deadline(5):
        assert post_json(client(handler), "https://x", {}, what="t") == {"ok": True}
    assert 0 < seen[0] <= 5  # not the client's 30 s


def test_retry_that_would_overrun_the_budget_fails_at_once(slept):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "20"})

    with deadline(5), pytest.raises(DeadlineExceeded, match="would pass the deadline"):
        post_json(client(handler), "https://x", {}, what="Groq chat request")
    assert calls == [1] and slept == []  # no pointless 20 s sleep


def test_spent_budget_stops_before_sending(slept):
    calls = []
    with deadline(-1), pytest.raises(DeadlineExceeded, match="used up"):
        post_json(client(lambda r: calls.append(1) or httpx.Response(200, json={})), "https://x", {}, what="t")
    assert calls == []


def test_nested_budgets_keep_the_earlier_end():
    with deadline(10):
        with deadline(100):
            assert remaining() <= 10
        with deadline(1):
            assert remaining() <= 1
    assert remaining() is None


def test_retries_still_work_inside_a_generous_budget(slept):
    replies = [httpx.Response(503), httpx.Response(200, json={"ok": 1})]
    with deadline(60):
        assert post_json(client(lambda r: replies.pop(0)), "https://x", {}, what="t") == {"ok": 1}
    assert slept == [1.0]


def test_a_daily_limit_fails_fast_instead_of_retrying(slept):
    from hyrag.http import QuotaExhausted
    calls = []
    body = {"error": {"message": "Rate limit reached for model `m` on tokens per day (TPD): Limit 200000, Used 199675. "
                                 "Please try again in 7m9.84s.", "code": "rate_limit_exceeded"}}

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json=body, headers={"Retry-After": "430"})

    with pytest.raises(QuotaExhausted, match="daily limit"):
        post_json(client(handler), "https://x", {}, what="Groq chat request")
    assert calls == [1] and slept == []  # one request, no pointless waits


def test_a_per_minute_limit_is_still_retried(slept):
    replies = [httpx.Response(429, json={"error": {"message": "on tokens per minute (TPM)"}}, headers={"Retry-After": "2"}),
               httpx.Response(200, json={"ok": 1})]
    assert post_json(client(lambda r: replies.pop(0)), "https://x", {}, what="t") == {"ok": 1}
    assert slept == [2.0]
