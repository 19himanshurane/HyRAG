import json

import pytest

from hyrag.citations import CitationReport, Claim, ClaimCheck
from hyrag.confidence import (CAPS, citation_coverage, assess_completeness, retrieval_confidence,
                              score_answer)
from hyrag.generation import NOT_FOUND, GroundedAnswer
from hyrag.llm import ChatResult

from .test_generation import hit


def passages(rerank=None, dense=None):
    h = hit("text")
    h.rerank_score, h.dense_score = rerank, dense
    return [h]


def report(*statuses):
    return CitationReport([ClaimCheck(Claim(f"c{i}", [1], "cited"), s) for i, s in enumerate(statuses)], 0)


class PartsJudge:
    def __init__(self, *statuses, raw=None):
        self.statuses, self.raw, self.calls = statuses, raw, 0

    def complete(self, messages, json_schema=None):
        self.calls += 1
        text = self.raw if self.raw is not None else json.dumps(
            {"parts": [{"part": f"p{i}", "status": s} for i, s in enumerate(self.statuses)]})
        return ChatResult(text, "fake", "stop", 1, 1, 0, 0.0)


# ----- retrieval: calibrated on eval/calibration_questions.json -----

@pytest.mark.parametrize("logit, low, high", [
    (9.4, 0.99, 1.0),    # clearest answerable question
    (0.08, 0.85, 0.95),  # lowest answerable (SQLSTATE 23505: a row in a big table)
    (-5.83, 0.05, 0.2),  # highest unanswerable (Slack password)
    (-11.25, 0.0, 0.01),
])
def test_retrieval_confidence_matches_the_calibration(logit, low, high):
    score, signal = retrieval_confidence(GroundedAnswer("q", "a", passages(rerank=logit)))
    assert low <= score <= high and signal == "rerank"


def test_retrieval_falls_back_to_dense_similarity_when_not_reranked():
    score, signal = retrieval_confidence(GroundedAnswer("q", "a", passages(dense=0.86)))
    assert signal == "dense" and score > 0.9
    assert retrieval_confidence(GroundedAnswer("q", "a", passages(dense=0.70)))[0] < 0.1
    assert retrieval_confidence(GroundedAnswer("q", "a", [])) == (0.0, "none")


# ----- citations and completeness -----

def test_citation_coverage_counts_partial_as_half_and_skips_gaps():
    r = report("supported", "partial", "unsupported", "uncited")
    r.checks.append(ClaimCheck(Claim("The documents do not cover pricing.", [], "gap"), "gap"))
    assert citation_coverage(r) == pytest.approx(1.5 / 4)
    assert citation_coverage(CitationReport([], 0)) == 0.0


def test_completeness_is_the_share_of_parts_answered():
    c = assess_completeness("Fix it, and what does it cost?", "Renew it [1]. Pricing isn't covered.",
                            PartsJudge("answered", "not_covered"))
    assert c.score == 0.5 and [p["status"] for p in c.parts] == ["answered", "not_covered"]


def test_unreadable_completeness_reply_counts_as_incomplete():
    c = assess_completeness("q", "a", PartsJudge(raw="not json"))
    assert c.score == 0.0 and c.error


# ----- the composite -----

def good_answer(**kw):
    return GroundedAnswer("q", "Renew it [1].", passages(rerank=8.0), citations=[1], **kw)


def test_fully_supported_complete_answer_is_high():
    c = score_answer(good_answer(), report("supported", "supported"), PartsJudge("answered"))
    assert c.level == "high" and c.score > 0.95 and c.flags == []


def test_any_unsupported_claim_keeps_the_answer_below_medium():
    # At the first cap (0.5) a deliberately wrong answer scored "medium" on the real corpus.
    c = score_answer(good_answer(), report("unsupported", "unsupported"), PartsJudge("answered"))
    assert c.level == "low" and c.score <= CAPS["unsupported_claim"] and "unsupported_claim" in c.flags


def test_a_partial_claim_rules_out_high():
    c = score_answer(good_answer(), report("supported", "partial"), PartsJudge("answered"))
    assert c.level != "high" and "partial_claim" in c.flags


@pytest.mark.parametrize("kw, flag", [(dict(ungrounded=True), "ungrounded"), (dict(truncated=True), "truncated")])
def test_red_flags_cap_the_score(kw, flag):
    c = score_answer(good_answer(**kw), report("supported"), PartsJudge("answered"))
    assert flag in c.flags and c.score <= CAPS[flag]


def test_ignoring_part_of_the_question_is_flagged():
    c = score_answer(good_answer(), report("supported"), PartsJudge("answered", "missing"))
    assert "missing_part" in c.flags and c.score <= CAPS["missing_part"]


def test_an_honest_half_answer_is_capped_at_medium_not_treated_as_a_red_flag():
    c = score_answer(good_answer(), report("supported"), PartsJudge("answered", "not_covered"))
    assert c.flags == ["incomplete"] and c.completeness == 0.5 and c.level == "medium"
    assert c.score > CAPS["missing_part"]  # milder than silently ignoring a part


def test_not_found_answer_is_scored_without_asking_the_judge():
    judge = PartsJudge("answered")
    a = GroundedAnswer("q", NOT_FOUND, passages(rerank=-9.0), insufficient=True)
    c = score_answer(a, CitationReport([], 0), judge)
    assert (c.level, c.score, judge.calls) == ("not_found", 0.0, 0) and c.retrieval < 0.05


def test_weaker_retrieval_signal_is_reported():
    a = GroundedAnswer("q", "Renew it [1].", passages(dense=0.9), citations=[1])
    assert "retrieval_from_dense" in score_answer(a, report("supported"), PartsJudge("answered")).flags
