"""One call from question to a structured, honest response: retrieve -> gate -> answer -> verify -> score.

Statuses (with a stable `code` a UI can translate):
- answered    answered                              claims verified (some sentences may still be flagged)
- partial     partial                               answers what the documents cover, names what they don't
- unverified  unverified.unsupported_claim|ungrounded   a claim is unsupported, or nothing is cited. The text
              is WITHHELD from `answer` and kept in `draft`; `verified_claims` lists the sentences that passed.
- unchecked   unchecked.checker_unavailable|timeout  the checking model failed or the time budget ran out, so
              the claims are neither verified nor refuted. Shown with a warning (show_unchecked=True) or withheld.
- not_found   not_found.no_relevant_documents|model_found_nothing   retrieval below the gate (the model is
              never called), or the model says the documents don't contain it
- error       error.retrieval_unavailable|writer_unavailable|timeout   no answer could be produced

Production behaviour (docs/phase3-audit.md):
- every question has a time budget (default 60 s) shared by all its network calls (hyrag.http.deadline);
- an outage never escapes as an exception: the writer being down is `error`, the judge being down is
  `unchecked` (not "unverified": an outage is not evidence that the answer is wrong), Mistral being down
  degrades retrieval to keyword search (`degraded`);
- each response carries a request id, per-step timings, requests and tokens, and is logged as one JSON line
  (the question itself is not logged, only its length and a hash: questions can contain personal data);
- an optional AnswerCache serves repeated questions until the index changes.

Why three layers rather than one retrieval threshold (eval/near_miss_questions.json): a near miss can score
HIGH retrieval: "How do I fix ERR_TUNNEL_4099?" (a code no document mentions) scored 0.90 because the
4012/4013 pages look relevant. There the writer said "not found" itself; when a writer does improvise,
citation verification is the layer that catches it.
"""
import dataclasses
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

from hyrag.citations import verify
from hyrag.confidence import Confidence, hit_relevance, passages_confidence, score_answer
from hyrag.generation import generate
from hyrag.http import DeadlineExceeded, QuotaExhausted, deadline
from hyrag.retrieval import FusedHit

log = logging.getLogger(__name__)

# Below this retrieval confidence the writer is not called. Calibration and near-miss sets (2026-09-26):
# every question that got a correct answer scored >= 0.78 (the lowest: a two-part question); unanswerable
# ones 0.00-0.34, except the near miss ERR_TUNNEL_4099 at 0.90 (why the gate is only the first layer).
# 0.5 = the rerank logit -3, the centre of the measured gap. Tuned on those 26 questions: re-check on the
# Phase 4 held-out set.
RETRIEVAL_GATE = 0.5
# A document is suggested for manual checking only if its best passage reaches this relevance. On the
# unanswerable questions it keeps SQLSTATE 23599 -> the PostgreSQL error-code appendix (0.34) and the
# Kubernetes pages for the autoscaler question (0.17), and drops everything for the printer (0.00).
SUGGEST_MIN_RELEVANCE = 0.1
CLOSEST_SHOWN = 3
# One question's total time. Measured: median 7.3 s, worst 40.5 s under rate limiting (18 live questions).
DEFAULT_BUDGET_SECONDS = 60.0


@dataclass
class SourceRef:
    source: str
    section: str
    page: int | None
    relevance: float | None       # 0..1, calibrated like retrieval confidence
    n: int | None = None          # the [n] the answer uses for it (cited sources only)


@dataclass
class Response:
    question: str
    status: str                   # answered | partial | unverified | unchecked | not_found | error
    code: str                     # stable machine-readable reason, e.g. "not_found.no_relevant_documents"
    message: str                  # one English line for the reader (translate by `code`)
    reason: str                   # details for logs and the dashboard
    answer: str | None = None     # shown for answered / partial (and unchecked when allowed)
    confidence: Confidence | None = None
    sources: list[SourceRef] = field(default_factory=list)       # what the answer cites, by [n]
    flagged_claims: list[str] = field(default_factory=list)      # sentences not to trust blindly
    verified_claims: list[str] = field(default_factory=list)     # sentences that passed (useful when withheld)
    not_covered: list[str] = field(default_factory=list)         # parts of the question the documents lack
    closest: list[SourceRef] = field(default_factory=list)       # what was found (not_found / unverified)
    check_manually: list[str] = field(default_factory=list)      # documents worth opening
    draft: str | None = None                      # model text withheld from `answer`
    degraded: list[str] = field(default_factory=list)            # parts that had to be skipped
    claim_counts: dict[str, int] = field(default_factory=dict)   # citation-check outcomes by status
    request_id: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)   # retrieve, generate, verify, score, total
    usage: dict[str, int] = field(default_factory=dict)          # requests and tokens per model
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class _Meter:
    """Counts one request's model calls and tokens (the clients are shared between requests)."""

    def __init__(self, chat):
        self.chat, self.requests, self.tokens = chat, 0, 0

    def complete(self, messages, json_schema=None):
        # The writer interface (generation.Chat) takes only messages: pass the schema only when there is one.
        result = (self.chat.complete(messages) if json_schema is None
                  else self.chat.complete(messages, json_schema=json_schema))
        self.requests += 1
        self.tokens += result.prompt_tokens + result.completion_tokens
        return result


class AnswerCache:
    """Recent responses by (question, index version): a repeated question costs no model call. Only outcomes
    that depend on the documents are kept (answered, partial, not_found), never errors or unchecked answers,
    which depend on an outage. Any write to the index changes its version, so stale answers are never served."""

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 3600.0):
        self.max_entries, self.ttl = max_entries, ttl_seconds
        self._items: OrderedDict[tuple, tuple[float, Response]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(question: str, index_version) -> tuple:
        return (" ".join(question.casefold().split()), index_version)

    def get(self, key) -> Response | None:
        with self._lock:
            item = self._items.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return item[1]

    def put(self, key, response: Response) -> None:
        if response.status not in ("answered", "partial", "not_found"):
            return
        with self._lock:
            self._items[key] = (time.monotonic(), response)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


def _ref(hit: FusedHit, n: int | None = None) -> SourceRef:
    c = hit.chunk
    relevance = hit_relevance(hit)
    return SourceRef(c.source, c.heading, c.page if c.page not in (None, -1) else None,
                     round(relevance, 3) if relevance is not None else None, n)


def _pointers(hits: list[FusedHit]) -> tuple[list[SourceRef], list[str]]:
    """What was found (the closest sections) and which documents are worth opening by hand."""
    related = [h for h in hits if (hit_relevance(h) or 0.0) >= SUGGEST_MIN_RELEVANCE]
    documents: list[str] = []
    for h in related:
        if h.chunk.source not in documents:
            documents.append(h.chunk.source)
    return [_ref(h) for h in related[:CLOSEST_SHOWN]], documents


def _suggest(documents: list[str]) -> str:
    return " These may be worth checking: " + ", ".join(documents) + "." if documents else ""


def _not_found(question: str, hits: list[FusedHit], code: str, reason: str) -> Response:
    closest, documents = _pointers(hits)
    message = "I couldn't find an answer to this in the documents." + (
        _suggest(documents) or " Nothing in the documents looks related.")
    return Response(question, "not_found", code, message, reason, closest=closest, check_manually=documents)


def ask(question: str, retriever, writer, judge, gate: float = RETRIEVAL_GATE,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS, cache: AnswerCache | None = None,
        show_unchecked: bool = True) -> Response:
    """Answer `question` from the indexed documents, or say honestly why not. Never raises for an outage or a
    timeout (those become `error` / `unchecked` responses); raises ValueError for a blank or over-long question.

    retriever: HybridRetriever (with a reranker); writer: the answering model; judge: the checking model."""
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    key = None
    if cache is not None:
        key = AnswerCache.key(question, retriever.index.version())
        hit = cache.get(key)
        if hit is not None:
            response = dataclasses.replace(hit, request_id=request_id, cached=True, usage={},
                                           timings_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})
            _log(response)
            return response

    w, j = _Meter(writer), _Meter(judge)
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - since) * 1000, 1)
        return now

    with deadline(budget_seconds):
        response = _run(question, retriever, w, j, gate, show_unchecked, lap, t0)
    response.request_id = request_id
    response.timings_ms = {**timings, "total": round((time.perf_counter() - t0) * 1000, 1)}
    response.usage = {"writer_requests": w.requests, "writer_tokens": w.tokens,
                      "judge_requests": j.requests, "judge_tokens": j.tokens}
    if cache is not None:
        cache.put(key, response)
    _log(response)
    return response


def _run(question, retriever, writer, judge, gate, show_unchecked, lap, t0) -> Response:
    try:
        retrieval = retriever.search(question)
    except ValueError:
        raise  # a bad question is the caller's to fix
    except Exception as e:
        # Logged with the stack trace: an outage and a programming bug both land here, and a bug must stay
        # visible to operators, not be buried under "unavailable".
        log.exception("retrieval failed")
        timeout = isinstance(e, DeadlineExceeded)
        return Response(question, "error", "error.timeout" if timeout else "error.retrieval_unavailable",
                        "Search is unavailable right now. Please try again shortly.",
                        f"retrieval failed: {type(e).__name__}: {str(e)[:200]}")
    since = lap("retrieve", t0)
    hits, degraded = retrieval.hits, retrieval.degraded

    relevance, signal = passages_confidence(hits)
    if relevance < gate:
        r = _not_found(question, hits, "not_found.no_relevant_documents",
                       f"retrieval confidence {relevance:.2f} < {gate} ({signal}); the model was not asked")
        r.degraded = degraded
        return r

    try:
        answer = generate(question, hits, writer)
    except Exception as e:
        log.exception("answer generation failed")
        code = ("error.timeout" if isinstance(e, DeadlineExceeded) else
                "error.quota_exhausted" if isinstance(e, QuotaExhausted) else "error.writer_unavailable")
        closest, documents = _pointers(hits)
        return Response(question, "error", code,
                        ("The daily usage limit of the answering service is reached." if code == "error.quota_exhausted"
                         else "The answering service is unavailable right now.") + _suggest(documents),
                        f"answer generation failed: {type(e).__name__}: {str(e)[:200]}",
                        closest=closest, check_manually=documents, degraded=degraded)
    since = lap("generate", since)
    report = verify(answer, judge)
    since = lap("verify", since)
    confidence = score_answer(answer, report, judge)
    lap("score", since)

    if answer.insufficient:
        r = _not_found(question, hits, "not_found.model_found_nothing",
                       "the model found no answer in the retrieved passages")
        r.confidence, r.degraded = confidence, degraded
        return r

    verified = [c.claim.text for c in report.checks if c.status == "supported"]
    counts: dict[str, int] = {}
    for c in report.checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    unsupported = [c.claim.text for c in report.checks if c.status == "unsupported"]
    if answer.ungrounded or unsupported:
        cited = [answer.passages[n - 1] for n in answer.citations] or hits
        closest, documents = _pointers(cited)
        return Response(
            question, "unverified",
            "unverified.ungrounded" if answer.ungrounded else "unverified.unsupported_claim",
            "I found related documents but couldn't verify an answer from them." + _suggest(documents),
            "the answer cites nothing" if answer.ungrounded else
            "citation check failed: " + "; ".join(unsupported)[:300],
            confidence=confidence, verified_claims=verified, closest=closest, check_manually=documents,
            draft=answer.text, degraded=degraded, claim_counts=counts)

    sources = [_ref(answer.passages[n - 1], n) for n in answer.citations]
    flagged = [c.claim.text for c in report.flagged]
    if any(c.status == "unchecked" for c in report.checks):
        timeout = any("DeadlineExceeded" in e for e in report.judge_errors)
        quota = any("QuotaExhausted" in e for e in report.judge_errors)
        return Response(
            question, "unchecked",
            "unchecked.quota_exhausted" if quota else "unchecked.timeout" if timeout else "unchecked.checker_unavailable",
            "This answer could not be checked against its sources" + (" in time" if timeout else
            " (the checking service is unavailable)") + ". Verify the cited sources before relying on it.",
            "judge failed: " + "; ".join(sorted(set(report.judge_errors)))[:300],
            answer=answer.text if show_unchecked else None, draft=None if show_unchecked else answer.text,
            confidence=confidence, sources=sources, flagged_claims=flagged, verified_claims=verified,
            degraded=degraded, claim_counts=counts)

    not_covered = [p["part"] for p in confidence.completeness_parts if p["status"] != "answered"]
    status = "partial" if not_covered else "answered"
    if status == "partial":
        message = "The documents answer only part of this question."
    elif confidence.level == "high":
        message = ""
    else:
        message = f"{confidence.level.capitalize()} confidence: check the cited sources."
    if degraded and "dense_search_unavailable" in degraded:
        message = (message + " " if message else "") + "(Search ran in keyword-only mode.)"
    return Response(question, status, status, message, f"confidence {confidence.score} ({confidence.level})",
                    answer=answer.text, confidence=confidence, sources=sources, flagged_claims=flagged,
                    verified_claims=verified, not_covered=not_covered, degraded=degraded, claim_counts=counts)


def _log(r: Response) -> None:
    """One JSON line per question. The question is not logged (it can contain personal data): only its
    length and a short hash, enough to spot repeats."""
    log.info(json.dumps({
        "event": "ask", "request_id": r.request_id, "status": r.status, "code": r.code, "cached": r.cached,
        "confidence": r.confidence.score if r.confidence else None, "degraded": r.degraded,
        "timings_ms": r.timings_ms, "usage": r.usage, "question_chars": len(r.question),
        "question_hash": hashlib.sha256(r.question.encode()).hexdigest()[:12],
    }))
