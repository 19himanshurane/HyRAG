"""Question limits for a public deployment: per visitor per hour, and a total per day."""
import pytest
from fastapi.testclient import TestClient

from hyrag.answer import AnswerCache
from hyrag.api import Settings, create_app
from hyrag.limits import DAY, HOUR, AskLimiter

from .test_answer import _Failing
from .test_api import services  # noqa: F401  (the fixture: real index + fake models)

Q = "How do I fix ERR_TUNNEL_4012 certificate expired?"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_visitor_is_stopped_at_the_hourly_limit_and_let_back_in_after_it():
    clock = Clock()
    limiter = AskLimiter(per_visitor_per_hour=2, clock=clock)
    for _ in range(2):
        assert limiter.check("ann") is None
        limiter.record("ann")
        clock.now += 60
    which, wait = limiter.check("ann")
    assert which == "visitor" and HOUR - 121 <= wait <= HOUR - 119  # until the first question is an hour old
    assert limiter.check("bob") is None  # someone else is unaffected
    clock.now += wait
    assert limiter.check("ann") is None


def test_the_daily_total_stops_everyone():
    clock = Clock()
    limiter = AskLimiter(per_visitor_per_hour=5, per_day=3, clock=clock)
    for visitor in ("ann", "bob", "cat"):
        limiter.record(visitor)
    assert limiter.check("dan")[0] == "daily"
    clock.now += DAY + 1
    assert limiter.check("dan") is None


def test_no_limits_means_nothing_is_ever_blocked_or_remembered():
    limiter = AskLimiter()
    for _ in range(500):
        limiter.record("ann")
    assert not limiter.enabled and limiter.check("ann") is None and limiter._visitors == {}


def ask(c, question=Q, visitor="ann"):
    return c.post("/v1/ask", json={"question": question}, headers={"X-Visitor-Id": visitor})


def test_api_returns_429_with_a_reason_and_when_to_come_back(services):  # noqa: F811
    with TestClient(create_app(services, Settings(visitor_limit=2))) as c:
        assert ask(c).status_code == 200 and ask(c, Q + " now").status_code == 200
        blocked = ask(c, Q + " again")
        assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 3000
        assert "2 questions per visitor per hour" in blocked.json()["detail"]
        assert ask(c, visitor="bob").status_code == 200
    assert len(services.writer.calls) == 3  # the blocked question never reached the model


def test_cached_answers_do_not_count_against_the_limit(services):  # noqa: F811
    services.cache = AnswerCache()
    with TestClient(create_app(services, Settings(visitor_limit=1))) as c:
        first = ask(c)
        repeats = [ask(c) for _ in range(3)]  # the same question again: served from the cache, free
        assert first.status_code == 200 and all(r.status_code == 200 and r.json()["cached"] for r in repeats)
        assert ask(c, Q + " really").status_code == 429  # a new question does cost one


def test_an_outage_does_not_use_up_a_visitors_questions(services):  # noqa: F811
    good = services.writer
    services.writer = _Failing(RuntimeError("Groq is down"))
    with TestClient(create_app(services, Settings(visitor_limit=1))) as c:
        assert ask(c).status_code == 503
        services.writer = good
        assert ask(c).status_code == 200


def test_the_daily_limit_message_differs_from_the_visitor_one(services):  # noqa: F811
    with TestClient(create_app(services, Settings(daily_limit=1))) as c:
        assert ask(c, visitor="ann").status_code == 200
        blocked = ask(c, Q + " too", visitor="bob")
    assert blocked.status_code == 429 and "budget for today" in blocked.json()["detail"]


@pytest.mark.parametrize("header", [{}, {"X-Visitor-Id": "x" * 500}])
def test_a_missing_or_huge_visitor_id_is_handled(services, header):  # noqa: F811
    with TestClient(create_app(services, Settings(visitor_limit=1))) as c:
        assert c.post("/v1/ask", json={"question": Q}, headers=header).status_code == 200
        assert c.post("/v1/ask", json={"question": Q + " 2"}, headers=header).status_code == 429
