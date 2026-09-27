import json

import pytest

from hyrag.answer import RETRIEVAL_GATE, SUGGEST_MIN_RELEVANCE, ask
from hyrag.citations import BATCH_SCHEMA, quote_in_passage
from hyrag.confidence import COMPLETENESS_SCHEMA
from hyrag.generation import NOT_FOUND
from hyrag.llm import ChatResult
from hyrag.retrieval import Retrieval

from .test_citations import parse_batch
from .test_generation import FakeChat, hit


def passage(text, source, logit, heading="Guide > Section"):
    h = hit(text, heading=heading, source=source)
    h.rerank_score = logit
    return h


class FakeRetriever:
    def __init__(self, hits, degraded=(), version=1):
        self.hits, self.degraded = hits, degraded
        self.index = type("Index", (), {"version": lambda _self: version})()

    def search(self, question):
        if not question.strip():
            raise ValueError("empty query")
        return Retrieval(list(self.hits), list(self.degraded))


class FakeJudge:
    """Citation verdicts by claim substring (batched format); completeness parts as given."""

    def __init__(self, verdicts=None, parts=("answered",), fail=None):
        self.verdicts, self.parts, self.fail, self.calls = verdicts or {}, parts, fail, 0

    def complete(self, messages, json_schema=None):
        self.calls += 1
        if self.fail:
            raise self.fail
        if json_schema is COMPLETENESS_SCHEMA:
            body = {"parts": [{"part": f"part {i}", "status": s} for i, s in enumerate(self.parts)]}
        else:
            assert json_schema is BATCH_SCHEMA
            passages, checks = parse_batch(messages[1]["content"])
            items = []
            for i, n, claim in checks:
                verdict = next((v for k, v in self.verdicts.items() if k in claim), "supported")
                items.append({"id": i, "quote": "" if verdict == "unsupported" else passages[n].split("\n\n")[-1],
                              "verdict": verdict, "reason": "scripted"})
            body = {"checks": items}
        return ChatResult(json.dumps(body), "fake", "stop", 1, 1, 0, 0.0)


VPN = passage("Renew the certificate in NimbusConnect.", "vpn-setup.md", 8.0, "VPN > Error 4012")
NIST = passage("Account recovery uses recovery codes.", "nist.pdf", -9.0)
PG = passage("23505 | unique_violation", "postgres-errors.html", -4.0)   # relevance ~0.33: weakly related


def test_low_retrieval_is_not_found_without_asking_the_model():
    writer, judge = FakeChat("should not be called"), FakeJudge()
    r = ask("What does SQLSTATE 23599 mean?", FakeRetriever([PG, NIST]), writer, judge)
    assert r.status == "not_found" and writer.calls == [] and judge.calls == 0
    assert r.answer is None and "model was not asked" in r.reason
    assert r.check_manually == ["postgres-errors.html"]      # weakly related: worth a look
    assert [c.source for c in r.closest] == ["postgres-errors.html"]  # NIST (relevance ~0) is not suggested
    assert "worth checking: postgres-errors.html" in r.message


def test_nothing_related_says_so_instead_of_suggesting_noise():
    r = ask("How do I set up the office printer?", FakeRetriever([NIST]), FakeChat("x"), FakeJudge())
    assert r.status == "not_found" and r.check_manually == [] and r.closest == []
    assert "Nothing in the documents looks related" in r.message


def test_model_saying_not_found_is_a_structured_not_found():
    r = ask("How do I fix ERR_TUNNEL_4099?", FakeRetriever([VPN]), FakeChat(NOT_FOUND), FakeJudge())
    assert r.status == "not_found" and r.check_manually == ["vpn-setup.md"] and r.answer is None
    assert "model found no answer" in r.reason


def test_unsupported_claim_withholds_the_answer():
    writer = FakeChat("ERR_TUNNEL_4012 means the gateway rejected your region [1].")
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), writer, FakeJudge({"gateway": "unsupported"}))
    assert r.status == "unverified" and r.answer is None
    assert r.draft.startswith("ERR_TUNNEL_4012 means the gateway")  # kept, but never presented as the answer
    assert "couldn't verify" in r.message and "gateway rejected" in r.reason
    assert r.confidence.level == "low"


def test_answer_citing_nothing_is_withheld():
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), FakeChat("Email your password to x."),
            FakeJudge())
    assert r.status == "unverified" and r.answer is None and "cites nothing" in r.reason


def test_verified_answer_lists_its_sources():
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN, NIST]), FakeChat("Renew it in NimbusConnect [1]."),
            FakeJudge())
    assert (r.status, r.message, r.answer) == ("answered", "", "Renew it in NimbusConnect [1].")
    assert [(s.n, s.source, s.section) for s in r.sources] == [(1, "vpn-setup.md", "VPN > Error 4012")]
    assert r.confidence.level == "high" and r.sources[0].relevance > 0.99


def test_half_answered_question_is_partial_and_names_the_gap():
    writer = FakeChat("Renew it [1]. The documents do not cover the monthly cost.")
    r = ask("Fix 4012, and what does the VPN cost?", FakeRetriever([VPN]), writer,
            FakeJudge(parts=("answered", "not_covered")))
    assert r.status == "partial" and r.answer and r.not_covered == ["part 1"]
    assert "only part" in r.message and r.confidence.level == "medium"


def test_bad_question_is_rejected_before_anything_runs():
    with pytest.raises(ValueError):
        ask("   ", FakeRetriever([VPN]), FakeChat("x"), FakeJudge())


def test_response_serialises_to_json():
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), FakeChat("Renew it [1]."), FakeJudge())
    data = json.loads(json.dumps(r.to_dict()))
    assert data["status"] == "answered" and data["sources"][0]["source"] == "vpn-setup.md"


def test_thresholds_are_consistent():
    assert 0 < SUGGEST_MIN_RELEVANCE < RETRIEVAL_GATE < 1


def test_quote_check_ignores_list_bullets_but_not_words():
    # A correct, supported answer was withheld because the judge quoted two bullet points without their "- ".
    p = "Three things to check:\n\n- Make sure the image name is correct.\n- Have you pushed the image?"
    assert quote_in_passage("Make sure the image name is correct. Have you pushed the image", p)
    assert not quote_in_passage("Make sure the image name is wrong", p)


# ----- Phase 3 production audit: outages, deadlines, observability, cache -----

from hyrag import http  # noqa: E402
from hyrag.answer import AnswerCache  # noqa: E402
from hyrag.http import DeadlineExceeded  # noqa: E402


class BrokenRetriever(FakeRetriever):
    def search(self, question):
        raise RuntimeError("index unavailable")


def test_writer_outage_is_a_structured_error_not_an_exception():
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]),
            _Failing(RuntimeError("Groq chat request failed after 3 attempts")), FakeJudge())
    assert (r.status, r.code) == ("error", "error.writer_unavailable") and r.answer is None
    assert r.check_manually == ["vpn-setup.md"]  # still points somewhere useful


def test_writer_timeout_is_reported_as_timeout():
    r = ask("q?", FakeRetriever([VPN]), _Failing(DeadlineExceeded("budget used up")), FakeJudge())
    assert r.code == "error.timeout"


def test_retrieval_outage_is_a_structured_error():
    r = ask("q?", BrokenRetriever([]), FakeChat("x"), FakeJudge())
    assert (r.status, r.code) == ("error", "error.retrieval_unavailable")


def test_judge_outage_is_unchecked_not_unverified():
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), FakeChat("Renew it [1]."),
            FakeJudge(fail=RuntimeError("judge down")))
    assert (r.status, r.code) == ("unchecked", "unchecked.checker_unavailable")
    assert r.answer == "Renew it [1]." and "could not be checked" in r.message and "judge down" in r.reason
    assert r.confidence.level == "low"
    hidden = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), FakeChat("Renew it [1]."),
                 FakeJudge(fail=RuntimeError("judge down")), show_unchecked=False)
    assert hidden.answer is None and hidden.draft == "Renew it [1]."


def test_judge_timeout_is_unchecked_timeout():
    r = ask("q?", FakeRetriever([VPN]), FakeChat("Renew it [1]."), FakeJudge(fail=DeadlineExceeded("budget")))
    assert r.code == "unchecked.timeout" and "in time" in r.message


def test_withheld_answer_still_lists_the_sentences_that_passed():
    writer = FakeChat("Renew it in NimbusConnect [1]. The gateway rejected your region [1].")
    r = ask("q?", FakeRetriever([VPN]), writer, FakeJudge({"gateway": "unsupported"}))
    assert r.code == "unverified.unsupported_claim" and r.verified_claims == ["Renew it in NimbusConnect."]


def test_degraded_search_is_reported():
    r = ask("q?", FakeRetriever([VPN], degraded=["dense_search_unavailable"]), FakeChat("Renew it [1]."), FakeJudge())
    assert r.status == "answered" and r.degraded == ["dense_search_unavailable"] and "keyword-only" in r.message


def test_response_carries_request_id_timings_and_usage(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="hyrag.answer")
    r = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), FakeChat("Renew it [1]."), FakeJudge())
    assert len(r.request_id) == 12
    assert {"retrieve", "generate", "verify", "score", "total"} <= set(r.timings_ms)
    assert r.usage == {"writer_requests": 1, "writer_tokens": 120, "judge_requests": 2, "judge_tokens": 4}
    line = json.loads(caplog.records[-1].getMessage())
    assert line["request_id"] == r.request_id and "ERR_TUNNEL" not in caplog.text  # the question isn't logged


def test_the_time_budget_reaches_the_model_calls():
    seen = []

    class Watching(FakeChat):
        def complete(self, messages):
            seen.append(http.remaining())
            return super().complete(messages)

    ask("q?", FakeRetriever([VPN]), Watching("Renew it [1]."), FakeJudge(), budget_seconds=7)
    assert seen and 0 < seen[0] <= 7
    assert http.remaining() is None  # the budget ends with the question


def test_cache_serves_repeats_until_the_index_changes():
    cache, writer = AnswerCache(), FakeChat("Renew it [1].")
    first = ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN]), writer, FakeJudge(), cache=cache)
    again = ask("  how do I fix err_tunnel_4012? ", FakeRetriever([VPN]), writer, FakeJudge(), cache=cache)
    assert again.cached and again.answer == first.answer and again.request_id != first.request_id
    assert len(writer.calls) == 1 and again.usage == {}
    ask("How do I fix ERR_TUNNEL_4012?", FakeRetriever([VPN], version=2), writer, FakeJudge(), cache=cache)
    assert len(writer.calls) == 2  # the index changed: asked again


def test_cache_never_keeps_outage_results():
    cache = AnswerCache()
    ask("q?", FakeRetriever([VPN]), _Failing(RuntimeError("down")), FakeJudge(), cache=cache)
    writer = FakeChat("Renew it [1].")
    assert ask("q?", FakeRetriever([VPN]), writer, FakeJudge(), cache=cache).status == "answered"
    assert len(writer.calls) == 1


class _Failing:
    def __init__(self, error):
        self.error = error

    def complete(self, messages):
        raise self.error


def test_daily_quota_exhaustion_has_its_own_codes():
    from hyrag.http import QuotaExhausted
    r = ask("q?", FakeRetriever([VPN]), _Failing(QuotaExhausted("daily limit reached")), FakeJudge())
    assert r.code == "error.quota_exhausted" and "daily usage limit" in r.message
    r = ask("q?", FakeRetriever([VPN]), FakeChat("Renew it [1]."), FakeJudge(fail=QuotaExhausted("daily limit reached")))
    assert r.code == "unchecked.quota_exhausted"
