"""ask(): question in, structured response out. Retrieve, gate, write, verify, score.

Statuses, each with a stable `code` a UI can translate:
- answered    answered                                  claims verified (some sentences may still be flagged)
- partial     partial                                   answers what the documents cover, says what they don't
- unverified  unverified.unsupported_claim|ungrounded   a claim isn't supported, or nothing is cited. The text is
              kept out of `answer` (it's in `draft`); `verified_claims` has the sentences that passed.
- unchecked   unchecked.checker_unavailable|timeout     the checker failed or time ran out, so nothing is known
              either way. Shown with a warning if show_unchecked, otherwise withheld.
- not_found   not_found.no_relevant_documents|model_found_nothing   below the retrieval gate (no model call), or
              the model found nothing in the passages
- error       error.retrieval_unavailable|writer_unavailable|timeout

Outages never raise. A writer outage is `error`; a checker outage is `unchecked`, not `unverified`, because an
outage says nothing about whether the answer is right; a Mistral outage falls back to keyword search. Each
question shares one time budget across its network calls, and is logged as one JSON line with the question
hashed, not stored, since questions can contain personal data.

The retrieval gate alone isn't enough: "How do I fix ERR_TUNNEL_4099?" (a code no document mentions) scores
0.90 because the 4012/4013 pages look relevant. The writer or the citation check has to catch those.
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
from hyrag.index import check_collection
from hyrag.retrieval import FusedHit

log = logging.getLogger(__name__)

# Below this the writer isn't called. Answerable calibration questions all scored 0.78 or more, unanswerable
# ones 0.34 or less (apart from the ERR_TUNNEL_4099 near miss above). 0.5 is rerank logit -3, the middle of
# that gap. Set on 26 questions, so it gets re-checked on the held-out evaluation set.
RETRIEVAL_GATE = 0.5
# Suggest a document to check by hand only above this. It keeps the PostgreSQL error-code appendix for an
# unknown SQLSTATE (0.34) and drops everything for the office printer (0.00).
SUGGEST_MIN_RELEVANCE = 0.1
CLOSEST_SHOWN = 3
# Median 7.3 s and worst 40.5 s under rate limiting, over 18 live questions.
DEFAULT_BUDGET_SECONDS = 60.0


@dataclass
class SourceRef:
    source: str
    section: str
    page: int | None
    relevance: float | None       # 0..1, calibrated like retrieval confidence
    n: int | None = None          # the [n] the answer uses for it (cited sources only)


@dataclass
class RetrievedPassage:
    """One passage retrieval handed to the writer, with how each search ranked it (for the dashboard)."""
    source: str
    section: str
    page: int | None
    text: str
    relevance: float | None       # 0..1, calibrated (see hyrag.confidence)
    rerank_score: float | None    # the reranker's raw logit
    fused_rank: int | None        # position after fusion, before reranking
    dense_rank: int | None        # position in meaning search (None: not found by it)
    sparse_rank: int | None       # position in keyword search (None: not found by it)


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
    search_queries: list[str] = field(default_factory=list)      # rewritten queries, when the question was rewritten
    mode: str = "hybrid"                                         # hybrid | dense | sparse retrieval
    collection: str | None = None                                # whose documents were searched (None = all)
    retrieved: list[RetrievedPassage] = field(default_factory=list)  # the passages the writer saw, ranked
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
    def key(question: str, index_version, mode: str = "hybrid", collection: str | None = None) -> tuple:
        # The collection is part of the key: one company's cached answer must never be served to another.
        return (" ".join(question.casefold().split()), index_version, mode, collection)

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


def _passage(hit: FusedHit) -> RetrievedPassage:
    c, relevance = hit.chunk, hit_relevance(hit)
    return RetrievedPassage(c.source, c.heading, c.page if c.page not in (None, -1) else None, c.text,
                            round(relevance, 3) if relevance is not None else None,
                            round(hit.rerank_score, 3) if hit.rerank_score is not None else None,
                            hit.fused_rank, hit.dense_rank, hit.sparse_rank)


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
        show_unchecked: bool = True, mode: str = "hybrid", collection: str | None = None) -> Response:
    """Answer `question` from the indexed documents, or say honestly why not. Never raises for an outage or a
    timeout (those become `error` / `unchecked` responses); raises ValueError for a blank or over-long question
    or an unknown collection.

    retriever: HybridRetriever (with a reranker); writer: the answering model; judge: the checking model.
    mode: "hybrid" (default), or "dense" / "sparse" to run one search alone (the dashboard's comparison).
    collection: answer from one company's documents only (hyrag.index.COLLECTIONS); None searches all."""
    check_collection(collection)
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    key = None
    if cache is not None:
        key = AnswerCache.key(question, retriever.index.version(), mode, collection)
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
        response = _run(question, retriever, w, j, gate, show_unchecked, lap, t0, mode, collection)
    response.request_id = request_id
    response.timings_ms = {**timings, "total": round((time.perf_counter() - t0) * 1000, 1)}
    response.usage = {**response.usage, "writer_requests": w.requests, "writer_tokens": w.tokens,
                      "judge_requests": j.requests, "judge_tokens": j.tokens}
    if cache is not None:
        cache.put(key, response)
    _log(response)
    return response


def _run(question, retriever, writer, judge, gate, show_unchecked, lap, t0, mode="hybrid",
         collection=None) -> Response:
    try:
        retrieval = retriever.search(question, mode=mode, collection=collection)
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
    response = _answer(question, retrieval, writer, judge, gate, show_unchecked, lap, since)
    response.search_queries = list(getattr(retrieval, "queries", []) or [])
    response.mode = mode
    response.collection = collection
    response.retrieved = [_passage(h) for h in retrieval.hits]
    response.usage = dict(getattr(retrieval, "usage", {}) or {})  # ask() adds the writer and judge counts
    return response


def _answer(question, retrieval, writer, judge, gate, show_unchecked, lap, since) -> Response:
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
