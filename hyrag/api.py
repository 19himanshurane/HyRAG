"""HTTP API: ask questions, list and add documents.

Run:  uvicorn hyrag.api:app --host 0.0.0.0 --port 8000     (OpenAPI docs at /docs)

Endpoints:
- POST /v1/ask         a question -> the structured answer (hyrag.answer.Response): answer, citations, confidence
                       by dimension, retrieved passages, status/code. Optional retrieval mode (hybrid | dense |
                       sparse) for the dashboard's comparison.
- GET  /v1/collections the document collections (one company's documents each) and their size.
- GET  /v1/documents   indexed documents with chunk counts (optionally one collection's).
- POST /v1/ingest      upload files to index into a collection (needs the admin key). Same name = a new version.
- GET  /health         liveness: the process is up.
- GET  /ready          readiness: keys, index and reranker model are usable (503 with the reasons if not).

Every answer the pipeline actually gives, including not_found and unverified, is a 200 with its status and
code: "the documents don't say" is a valid result, not a server failure. Only `error` becomes a 5xx: 503 for
outages and exhausted quota, 504 for timeouts.

Configuration (environment):
  HYRAG_DATA_DIR (data)  HYRAG_STRATEGY (structure)  HYRAG_BUDGET_SECONDS (60)
  HYRAG_API_KEY          if set, /v1/ask and /v1/documents require header X-API-Key
  HYRAG_ADMIN_KEY        required for /v1/ingest; unset = ingestion disabled
  HYRAG_CORS_ORIGINS     comma-separated origins allowed to call the API from a browser (the dashboard)
  HYRAG_MAX_UPLOAD_MB    per-file upload limit (default 20, never above the loader's 50)
  HYRAG_CHROMA_URL       a Chroma server (e.g. http://chroma:8000); unset = embedded store in HYRAG_DATA_DIR
"""
import hmac
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, Response as HTTPResponse, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hyrag.answer import AnswerCache, DEFAULT_BUDGET_SECONDS, Response, ask
from hyrag.index import COLLECTIONS, DEFAULT_COLLECTION, collection_of
from hyrag.loader import MAX_FILE_BYTES, SUPPORTED
from hyrag.retrieval import MAX_QUERY_CHARS

log = logging.getLogger(__name__)

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


@dataclass
class Settings:
    data_dir: Path = Path("data")
    strategy: str = "structure"
    budget_seconds: float = DEFAULT_BUDGET_SECONDS
    api_key: str = ""
    admin_key: str = ""
    cors_origins: list[str] = field(default_factory=list)
    max_upload_bytes: int = 20 * 1024 * 1024
    chroma_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        mb = float(os.environ.get("HYRAG_MAX_UPLOAD_MB", "20"))
        return cls(data_dir=Path(os.environ.get("HYRAG_DATA_DIR", "data")),
                   strategy=os.environ.get("HYRAG_STRATEGY", "structure"),
                   budget_seconds=float(os.environ.get("HYRAG_BUDGET_SECONDS", DEFAULT_BUDGET_SECONDS)),
                   api_key=os.environ.get("HYRAG_API_KEY", ""), admin_key=os.environ.get("HYRAG_ADMIN_KEY", ""),
                   cors_origins=[o.strip() for o in os.environ.get("HYRAG_CORS_ORIGINS", "").split(",") if o.strip()],
                   max_upload_bytes=min(int(mb * 1024 * 1024), MAX_FILE_BYTES),
                   chroma_url=os.environ.get("HYRAG_CHROMA_URL", ""))


@dataclass
class Services:
    """Everything a request needs, built once at startup (the reranker takes ~10 s to load)."""
    retriever: object
    writer: object
    judge: object
    pipeline: object | None = None       # None: ingestion unavailable
    cache: AnswerCache | None = None
    problems: list[str] = field(default_factory=list)  # why the service is not ready (empty = ready)
    closers: list = field(default_factory=list)
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)  # one upload batch at a time

    @property
    def index(self):
        return self.retriever.index


def build_services(settings: Settings) -> Services:
    """Open the index, load and warm the reranker, connect the models. Missing pieces are recorded as
    readiness problems (served as 503 by /ready) instead of crashing the process at startup."""
    from dotenv import load_dotenv

    from hyrag.citations import judge_client
    from hyrag.embeddings import MistralEmbedder
    from hyrag.index import ChunkIndex
    from hyrag.llm import GroqChat
    from hyrag.loader import DocumentStore
    from hyrag.pipeline import Pipeline
    from hyrag.rerank import CrossEncoderScorer, is_downloaded
    from hyrag.retrieval import HybridRetriever

    load_dotenv()
    problems = [f"{k} is not set" for k in ("MISTRAL_API_KEY", "GROQ_API_KEY") if not os.environ.get(k)]
    if problems:
        raise RuntimeError("; ".join(problems))  # nothing can work without the keys: fail loudly at startup
    embedder = MistralEmbedder(cache_path=settings.data_dir / "embeddings.sqlite")  # with the data, not the cwd
    writer, judge = GroqChat(), judge_client()
    index = ChunkIndex(embedder, data_dir=settings.data_dir / "index", strategy=settings.strategy,
                       chroma_url=settings.chroma_url or None)
    if index.count() == 0:
        problems.append(f"the {settings.strategy!r} index is empty: run the seed/ingest step")
    else:
        started = time.perf_counter()
        index.warm()  # otherwise the first question loads everything and can run out of time
        log.info("search data loaded in %.1f s", time.perf_counter() - started)
    scorer = CrossEncoderScorer()
    if not is_downloaded():
        problems.append("reranker model not downloaded: python -m hyrag.rerank")
    else:
        scorer.score("warm up", ["load the model now, not on the first request"])
    retriever = HybridRetriever(index, reranker=scorer, rewriter=writer)
    pipeline = Pipeline(DocumentStore(settings.data_dir), index)
    return Services(retriever, writer, judge, pipeline, AnswerCache(), problems,
                    closers=[embedder.close, writer.close, judge.close])


# ----- request / response schemas (the OpenAPI contract) -----

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, examples=["How do I fix ERR_TUNNEL_4012?"])
    mode: Literal["hybrid", "dense", "sparse"] = Field(
        "hybrid", description="Retrieval mode; dense/sparse run one search alone, for comparison.")
    collection: Literal[*COLLECTIONS] | None = Field(
        None, description="Answer from one company's documents only (see GET /v1/collections). "
                          "Omitted: all documents, which can mix different companies' policies.")


class CollectionInfo(BaseModel):
    name: str
    documents: int
    chunks: int


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int
    duplicates: int = Field(description="chunks skipped as near-duplicates of text indexed elsewhere")


class DocumentList(BaseModel):
    total: int
    documents: list[DocumentInfo]


class IngestedFile(BaseModel):
    filename: str
    status: Literal["indexed", "failed"]
    doc_id: str | None = None
    chunks: int | None = None
    error: str | None = None


class IngestResult(BaseModel):
    files: list[IngestedFile]


# ----- the app -----

ERROR_HTTP = {"error.quota_exhausted": 503, "error.timeout": 504}


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    """`services` given (tests): used as is. Otherwise they are built at startup from the environment."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(settings)
        yield
        for close in app.state.services.closers:
            try:
                close()
            except Exception:
                log.exception("closing a client failed")

    app = FastAPI(title="HyRAG", version="1.0.0", lifespan=lifespan,
                  description="Hybrid-search question answering over internal documents, with verified citations "
                              "and an honest 'not found'.")
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key"])

    def svc(request: Request) -> Services:
        return request.app.state.services

    def require_key(x_api_key: str = Header("", alias="X-API-Key")) -> None:
        if settings.api_key and not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(401, "missing or wrong X-API-Key")

    def require_admin(x_api_key: str = Header("", alias="X-API-Key")) -> None:
        if not settings.admin_key:
            raise HTTPException(403, "ingestion is disabled (HYRAG_ADMIN_KEY is not set)")
        if not hmac.compare_digest(x_api_key, settings.admin_key):
            raise HTTPException(401, "missing or wrong X-API-Key")

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(response: HTTPResponse, services: Services = Depends(svc)) -> dict:
        if services.problems:
            response.status_code = 503
        return {"ready": not services.problems, "problems": services.problems,
                "chunks": services.index.count()}

    @app.post("/v1/ask", response_model=Response, tags=["questions"], dependencies=[Depends(require_key)],
              responses={503: {"description": "service or daily quota unavailable (body is the same schema)"},
                         504: {"description": "time budget exceeded (body is the same schema)"}})
    def ask_question(body: AskRequest, response: HTTPResponse, services: Services = Depends(svc)) -> Response:
        """Answer from the indexed documents. `status` says what happened (answered, partial, unverified, unchecked,
        not_found, error) and `code` why; `confidence` has the score by dimension; `sources` are the cited
        passages by [n]; `retrieved` lists every passage the writer saw, with how each search ranked it."""
        try:
            result = ask(body.question, services.retriever, services.writer, services.judge,
                         budget_seconds=settings.budget_seconds, cache=services.cache, mode=body.mode,
                         collection=body.collection)
        except ValueError as e:  # blank or too long (after trimming): the caller's to fix
            raise HTTPException(422, str(e))
        response.headers["X-Request-ID"] = result.request_id
        if result.status == "error":
            response.status_code = ERROR_HTTP.get(result.code, 503)
            if result.code == "error.quota_exhausted":
                response.headers["Retry-After"] = "3600"
        return result

    def document_rows(services: Services) -> list[tuple]:
        db = services.index.db
        with services.index._lock:  # the index's own lock: SQLite is shared with other requests
            return db.execute(
                "SELECT d.doc_id, d.source, COALESCE(c.n, 0), COALESCE(x.n, 0) FROM "
                "(SELECT doc_id, source FROM chunks UNION SELECT doc_id, source FROM duplicates) d "
                "LEFT JOIN (SELECT doc_id, COUNT(*) n FROM chunks GROUP BY doc_id) c ON c.doc_id = d.doc_id "
                "LEFT JOIN (SELECT doc_id, COUNT(*) n FROM duplicates GROUP BY doc_id) x ON x.doc_id = d.doc_id "
                "GROUP BY d.doc_id ORDER BY d.source").fetchall()

    @app.get("/v1/collections", response_model=list[CollectionInfo], tags=["documents"],
             dependencies=[Depends(require_key)])
    def collections(services: Services = Depends(svc)) -> list[CollectionInfo]:
        """The document collections (one company's documents each) and their size. Pass a name as
        `collection` to /v1/ask to answer from that collection only."""
        rows = document_rows(services)
        return [CollectionInfo(name=name, documents=sum(collection_of(r[1]) == name for r in rows),
                               chunks=sum(r[2] for r in rows if collection_of(r[1]) == name))
                for name in COLLECTIONS]

    @app.get("/v1/documents", response_model=DocumentList, tags=["documents"], dependencies=[Depends(require_key)])
    def documents(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                  collection: Literal[*COLLECTIONS] | None = None,
                  services: Services = Depends(svc)) -> DocumentList:
        """Documents in the index, with how many chunks each contributes (optionally one collection's only)."""
        rows = [r for r in document_rows(services) if collection is None or collection_of(r[1]) == collection]
        page = rows[offset:offset + limit]
        return DocumentList(total=len(rows), documents=[DocumentInfo(doc_id=r[0], source=r[1], chunks=r[2],
                                                                     duplicates=r[3]) for r in page])

    @app.post("/v1/ingest", response_model=IngestResult, tags=["documents"], dependencies=[Depends(require_admin)])
    def ingest(files: list[UploadFile] = File(...), collection: Literal[*COLLECTIONS] = DEFAULT_COLLECTION,
               services: Services = Depends(svc)) -> IngestResult:
        """Parse, chunk and index uploaded files (.md .mdx .txt .html .pdf) into a collection (default: demo).
        A file with the same name in the same collection replaces it. Each file succeeds or fails on its own."""
        if services.pipeline is None:
            raise HTTPException(503, "ingestion is unavailable")
        results: list[IngestedFile] = []
        with tempfile.TemporaryDirectory(prefix="hyrag-upload-") as root:
            # A named collection is the document's first folder ("posthog/x.md"): that is how it is told apart.
            tmp = Path(root) / collection if collection != DEFAULT_COLLECTION else Path(root)
            tmp.mkdir(exist_ok=True)
            accepted: list[Path] = []
            for upload in files:
                name = Path(upload.filename or "").name
                if not SAFE_NAME.match(name) or Path(name).suffix.lower() not in SUPPORTED:
                    results.append(IngestedFile(filename=name or "?", status="failed",
                                                error=f"name must be simple (letters, digits, . _ -) and end in one of "
                                                      f"{sorted(SUPPORTED)}"))
                    continue
                data = upload.file.read(settings.max_upload_bytes + 1)
                if len(data) > settings.max_upload_bytes:
                    results.append(IngestedFile(filename=name, status="failed",
                                                error=f"larger than {settings.max_upload_bytes // (1024 * 1024)} MB"))
                    continue
                path = tmp / name
                path.write_bytes(data)
                accepted.append(path)
            with services.ingest_lock:  # uploads are rare and heavy: one batch at a time
                for path in accepted:
                    try:
                        doc = services.pipeline.ingest_one(path, root=Path(root))  # source keeps its collection folder
                    except Exception as e:  # this file fails; the others still go in
                        log.warning("ingest of %s failed: %s", path.name, e)
                        results.append(IngestedFile(filename=path.name, status="failed",
                                                    error=f"{type(e).__name__}: {str(e)[:250]}"))
                        continue
                    with services.index._lock:
                        n = services.index.db.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?",
                                                      (doc.doc_id,)).fetchone()[0]
                    results.append(IngestedFile(filename=path.name, status="indexed", doc_id=doc.doc_id, chunks=n))
        return IngestResult(files=results)

    return app


app = create_app()  # `uvicorn hyrag.api:app`: cheap to import; services are built when the server starts
