"""
FastAPI 백엔드 서버
- /search: 하이브리드 검색
- /graph/stats: 그래프 통계
- /vector/stats: 벡터 통계
- /ingest: 데이터 수집 트리거
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from graph.graph_builder import GraphBuilder
from vectordb.vector_store import VectorStore
from search.hybrid_search import HybridSearchEngine

# ── 앱 초기화 ─────────────────────────────────────────────────────
app = FastAPI(
    title="BA Graph RAG API",
    description="Brity Automation 지식 검색 API — Neo4j + ChromaDB 하이브리드",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스
graph: GraphBuilder = None
vector: VectorStore = None
engine: HybridSearchEngine = None


@app.on_event("startup")
async def startup():
    global graph, vector, engine
    logger.info("서버 시작 — DB 연결 초기화")
    graph = GraphBuilder()
    vector = VectorStore()
    engine = HybridSearchEngine(graph, vector)

    # 데모 데이터 로드 (벡터 DB가 비어있으면)
    stats = vector.get_stats()
    if stats.get("ba_unified", 0) == 0:
        logger.info("벡터 DB 비어있음 — 데모 데이터 로드")
        vector.load_demo_data()

    logger.info("서버 초기화 완료")


# ── 요청/응답 모델 ────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    collection: Optional[str] = "ba_unified"
    top_k: Optional[int] = 5


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    graph_results: list[str]
    vector_results: list[dict]
    fused_contexts: list[dict]
    error_codes: list[str]
    processing_steps: list[str]
    confidence: float


class IngestRequest(BaseModel):
    source_type: str   # "html_all" | "pdf_basic" | "pdf_advanced" | "all"
    force_refresh: bool = False


# ── 엔드포인트 ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "BA Graph RAG API"}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not engine:
        raise HTTPException(503, "검색 엔진 초기화 중")
    if not req.query.strip():
        raise HTTPException(400, "질의가 비어있습니다")

    result = engine.search(req.query)
    return SearchResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        graph_results=result.graph_results,
        vector_results=result.vector_results,
        fused_contexts=result.fused_contexts,
        error_codes=result.error_codes_found,
        processing_steps=result.processing_steps,
        confidence=result.confidence,
    )


@app.get("/stats")
async def stats():
    g_stats = graph.get_stats() if graph else {}
    v_stats = vector.get_stats() if vector else {}
    return {
        "graph": g_stats,
        "vector": v_stats,
        "llm_enabled": engine._use_llm if engine else False,
    }


@app.post("/ingest")
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    bg.add_task(_run_ingest, req.source_type, req.force_refresh)
    return {"status": "started", "source_type": req.source_type}


async def _run_ingest(source_type: str, force: bool):
    """백그라운드 데이터 수집"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from crawler.html_crawler import BAManualCrawler
    from parser.pdf_parser import PDFParser

    crawler = BAManualCrawler()
    parser = PDFParser()

    if source_type in ("html_all", "all"):
        all_sections = crawler.crawl_all()
        for key, sections in all_sections.items():
            graph.build_from_html_sections(sections, key)
            for sec in sections:
                vector.add_html_section(sec, key)

    if source_type in ("pdf_basic", "all"):
        chunks = parser.parse("data/raw/BA기본과정_기본기능 및 실습 교육교재.pdf", "training_basic")
        graph.build_from_pdf_chunks(chunks, "training_basic")
        for chunk in chunks:
            vector.add_pdf_chunk(chunk, "training_basic")

    if source_type in ("pdf_advanced", "all"):
        chunks = parser.parse("data/raw/BA심화과정_E2E구축방법교육교재.pdf", "training_advanced")
        graph.build_from_pdf_chunks(chunks, "training_advanced")
        for chunk in chunks:
            vector.add_pdf_chunk(chunk, "training_advanced")

    logger.info(f"수집 완료: {source_type}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
