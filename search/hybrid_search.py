"""
Graph RAG 하이브리드 검색 엔진
- Step 1: 쿼리 분석 (엔티티 추출)
- Step 2: Neo4j 그래프 탐색
- Step 3: ChromaDB 벡터 검색
- Step 4: RRF Fusion
- Step 5: 교차 소스 병합
- Step 6: LLM 답변 생성
- Step 7: 출처 포맷팅
"""
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
import openai

try:
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False


@dataclass
class SearchResult:
    """최종 검색 결과"""
    query: str
    answer: str
    graph_results: list[dict]
    vector_results: list[dict]
    fused_contexts: list[dict]
    sources: list[dict]
    error_codes_found: list[str]
    processing_steps: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class FusedContext:
    content: str
    source_doc: str
    source_url: str
    title: str
    rrf_score: float
    origin: str  # "graph" | "vector" | "both"
    metadata: dict = field(default_factory=dict)


class HybridSearchEngine:
    """Graph RAG 하이브리드 검색 엔진"""

    # ── BA 도메인 키워드 ──────────────────────────────────────────
    FEATURE_KEYWORDS = [
        "스케줄러", "봇", "워크플로우", "프로세스", "태스크", "액션",
        "트리거", "에이전트", "모니터링", "배포", "로그", "백업",
        "사용자 관리", "권한", "설치", "BA Designer", "BA Runner",
        "BA Server", "BA Manager", "오케스트레이터",
    ]
    ERROR_PATTERN = re.compile(r"ERR-\d{4}")

    def __init__(self, graph_builder, vector_store):
        self.graph = graph_builder
        self.vector = vector_store
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.graph_top_k = int(os.getenv("GRAPH_TOP_K", "10"))
        self.vector_top_k = int(os.getenv("VECTOR_TOP_K", "15"))
        self.final_top_k = int(os.getenv("FINAL_TOP_K", "5"))

        # LLM 클라이언트
        if OPENAI_OK and os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
            self.llm = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self._use_llm = True
        else:
            self.llm = None
            self._use_llm = False
            logger.info("LLM 미연결 — 템플릿 답변 모드")

    # ── Step 1: 쿼리 분석 ────────────────────────────────────────
    def _analyze_query(self, query: str) -> dict:
        features = [kw for kw in self.FEATURE_KEYWORDS if kw in query]
        error_codes = self.ERROR_PATTERN.findall(query)

        # 의도 분류
        if any(w in query for w in ["오류", "에러", "ERR", "문제", "실패", "안 됨", "안됨"]):
            intent = "error_resolution"
        elif any(w in query for w in ["설치", "설정", "구성", "환경"]):
            intent = "install_config"
        elif any(w in query for w in ["어떻게", "방법", "절차", "단계", "순서"]):
            intent = "how_to"
        elif any(w in query for w in ["무엇", "이란", "개념", "설명"]):
            intent = "concept"
        else:
            intent = "general"

        return {
            "features": features,
            "error_codes": error_codes,
            "intent": intent,
            "keywords": query.split(),
        }

    # ── Step 2: 그래프 탐색 ──────────────────────────────────────
    def _graph_search(self, analysis: dict) -> list[dict]:
        results = []

        for feat in analysis["features"]:
            try:
                r = self.graph.search_by_feature_simple(feat)
                if r:
                    results.extend(r)
            except Exception as e:
                logger.debug(f"그래프 탐색 오류 ({feat}): {e}")

        for code in analysis["error_codes"]:
            try:
                r = self.graph.search_error(code)
                if r:
                    results.extend(r)
            except Exception as e:
                logger.debug(f"오류 코드 탐색 오류 ({code}): {e}")

        return results[:self.graph_top_k]

    # ── Step 3: 벡터 검색 ────────────────────────────────────────
    def _vector_search(self, query: str, analysis: dict) -> list:
        # 통합 컬렉션 검색
        results = self.vector.search(query, collection="ba_unified",
                                     n_results=self.vector_top_k)

        # 오류 코드가 있으면 장애조치 컬렉션 추가 검색
        if analysis["error_codes"]:
            ts_results = self.vector.search(
                query, collection="ba_troubleshoot", n_results=5
            )
            results = results + ts_results

        # 설치/설정 의도이면 설치매뉴얼 추가
        if analysis["intent"] == "install_config":
            inst_results = self.vector.search(
                query, collection="ba_install_manual", n_results=5
            )
            results = results + inst_results

        return results[:self.vector_top_k + 5]

    # ── Step 4: RRF Fusion ────────────────────────────────────────
    def _rrf_fusion(self, graph_results: list, vector_results: list, k: int = None) -> list[FusedContext]:
        if k is None:
            k = self.rrf_k
        scores: dict[str, dict] = {}

        # 그래프 결과 점수화
        for rank, item in enumerate(graph_results):
            # 그래프 결과를 텍스트로 변환 (안전 처리)
            try:
                content = self._graph_item_to_text(item)
            except Exception as e:
                logger.debug(f"_graph_item_to_text 오류, 항목을 문자열로 처리합니다: {e}")
                content = str(item)
            # If item is not a dict (e.g., plain string), avoid calling .get on it
            item_map = item if isinstance(item, dict) else {"_raw": item}
            url = str(item_map.get("feature_url") or item_map.get("url") or "")
            key = url or content[:80]

            if key not in scores:
                scores[key] = {
                    "content": content,
                    "source_doc": str(item_map.get("feature") or item_map.get("code") or ""),
                    "url": url,
                    "title": str(item_map.get("feature") or item_map.get("title") or ""),
                    "graph_score": 0.0,
                    "vector_score": 0.0,
                    "origin": "graph",
                    "metadata": item if isinstance(item, dict) else {"raw": item},
                }
            scores[key]["graph_score"] += 1.0 / (k + rank + 1)

        # 벡터 결과 점수화
        for rank, item in enumerate(vector_results):
            key = item.source_url or item.content[:80]

            if key not in scores:
                scores[key] = {
                    "content": item.content,
                    "source_doc": item.source_doc,
                    "url": item.source_url,
                    "title": item.metadata.get("title", ""),
                    "graph_score": 0.0,
                    "vector_score": 0.0,
                    "origin": "vector",
                    "metadata": item.metadata,
                }
            else:
                scores[key]["origin"] = "both"
            scores[key]["vector_score"] += 1.0 / (k + rank + 1)

        # RRF 합산 및 정렬
        fused = []
        for key, v in scores.items():
            rrf_score = v["graph_score"] + v["vector_score"]
            fused.append(FusedContext(
                content=v["content"],
                source_doc=v["source_doc"],
                source_url=v["url"],
                title=v["title"],
                rrf_score=rrf_score,
                origin=v["origin"],
                metadata=v["metadata"],
            ))

        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused  # 상위 N 개 선택은 search() 에서 처리

    def _graph_item_to_text(self, item: dict) -> str:
        parts = []
        # If item is not a dict (string, object, etc.), treat it as plain text
        if not isinstance(item, dict):
            return str(item)

        if item.get("feature"):
            parts.append(f"기능: {item['feature']}")
        if item.get("code"):
            parts.append(f"오류코드: {item['code']}")
            if item.get("content"):
                parts.append(item["content"])

        # resolutions: accept str, list[str], list[dict]
        res = item.get("resolutions")
        if res:
            if isinstance(res, str):
                parts.append(f"해결: {res}")
            else:
                for r in (res or []):
                    if isinstance(r, dict):
                        content = r.get("content") or r.get("text")
                        if content:
                            parts.append(f"해결: {content}")
                    elif r:
                        parts.append(f"해결: {str(r)}")

        # scenarios: accept str, list[str], list[dict]
        sc = item.get("scenarios")
        if sc:
            if isinstance(sc, str):
                parts.append(f"실습: {sc}")
            else:
                for s in (sc or []):
                    if isinstance(s, dict):
                        title = s.get("title") or s.get("name")
                        if title:
                            parts.append(f"실습: {title}")
                    elif s:
                        parts.append(f"실습: {str(s)}")

        # faqs: accept dict, str, list
        faqs = item.get("faqs")
        if faqs:
            # normalize single-dict FAQ to list
            if isinstance(faqs, dict):
                faqs = [faqs]
            if isinstance(faqs, str):
                parts.append(f"FAQ Q: {faqs}")
            else:
                for f in (faqs or []):
                    if isinstance(f, dict):
                        q = f.get("q") or f.get("question")
                        a = f.get("a") or f.get("answer")
                        if q:
                            parts.append(f"FAQ Q: {q}")
                            if a:
                                parts.append(f"FAQ A: {a}")
                    elif f:
                        parts.append(f"FAQ Q: {str(f)}")

        return "\n".join(parts) if parts else str(item)

    # ── Step 5 & 6: 컨텍스트 빌드 + LLM 답변 생성 ────────────────
    def _build_context(self, fused: list[FusedContext]) -> str:
        parts = []
        for i, ctx in enumerate(fused, 1):
            src = f"[출처 {i}: {ctx.source_doc}]"
            if ctx.source_url:
                src += f" ({ctx.source_url})"
            parts.append(f"{src}\n제목: {ctx.title}\n내용: {ctx.content}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, query: str, context: str, analysis: dict) -> str:
        if self._use_llm:
            return self._llm_answer(query, context, analysis)
        return self._template_answer(query, context, analysis)

    def _llm_answer(self, query: str, context: str, analysis: dict) -> str:
        system_prompt = """당신은 Brity Automation(Samsung SDS RPA 플랫폼) 전문 기술 지원 어시스턴트입니다.
아래 컨텍스트(검색된 문서 내용)를 기반으로 사용자 질문에 정확하고 친절하게 답변하세요.

규칙:
1. 컨텍스트에 없는 내용을 임의로 생성하지 마세요.
2. 답변 마지막에 참조한 출처를 명시하세요.
3. 오류 해결 시 단계별로 명확하게 안내하세요.
4. 한국어로 답변하세요."""

        user_msg = f"""질문: {query}

검색된 관련 문서:
{context}

위 내용을 바탕으로 질문에 답변해주세요."""

        try:
            resp = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 호출 오류: {e}")
            return self._template_answer(query, context, analysis)

    def _template_answer(self, query: str, context: str, analysis: dict) -> str:
        """LLM 없을 때 템플릿 기반 답변"""
        lines = ["**Brity Automation 지식 검색 결과(LLM 미연결)**\n"]

        if analysis["error_codes"]:
            lines.append(f"🔴 **감지된 오류 코드**: {', '.join(analysis['error_codes'])}\n")

        if analysis["features"]:
            lines.append(f"📌 **관련 기능**: {', '.join(analysis['features'])}\n")

        lines.append("---\n**검색된 관련 내용:**\n")

        # 컨텍스트에서 핵심 문장 추출
        for block in context.split("---"):
            if block.strip():
                src_line = [l for l in block.split("\n") if l.startswith("[출처")]
                content_line = [l for l in block.split("\n") if l.startswith("내용:")]
                if src_line and content_line:
                    lines.append(f"{src_line[0]}")
                    content = content_line[0].replace("내용:", "").strip()
                    lines.append(f"{content[:300]}{'...' if len(content) > 300 else ''}\n")

        lines.append("\n*💡 더 정확한 답변을 위해 OpenAI API 키를 설정하면 AI 답변을 받을 수 있습니다.*")
        return "\n".join(lines)

    # ── 메인 검색 함수 (Hybrid: RRF Fusion) ────────────────────────────────────
    def search(self, query: str, rrf_k: int = None, top_k: int = None) -> SearchResult:
        return self._search_hybrid(query, rrf_k, top_k)

    def _search_hybrid(self, query: str, rrf_k: int = None, top_k: int = None) -> SearchResult:
        """하이브리드 검색 (그래프 + 벡터 RRF 통합)"""
        steps = []

        # RRF k 값 설정 (전달된 값 우선)
        if rrf_k is not None:
            logger.info(f"RRF k 값: {rrf_k} (UI 설정)")
        else:
            logger.info(f"RRF k 값: {self.rrf_k} (기본값)")


        # 결과 개수 설정 (전달된 값 우선)
        if top_k is not None:
            logger.info(f"결과 개수: {top_k} (UI 설정)")
            final_top_k = top_k
        else:
            logger.info(f"결과 개수: {self.final_top_k} (기본값)")
            final_top_k = self.final_top_k

        # Step 1
        steps.append("✅ Step 1: 쿼리 분석 (엔티티·의도 추출)")
        analysis = self._analyze_query(query)
        logger.info(f"쿼리 분석: {analysis}")

        # Step 2
        steps.append("✅ Step 2: Neo4j 그래프 탐색")
        graph_results = self._graph_search(analysis)
        steps.append(f"   → 그래프 결과 {len(graph_results)}개")

        # Step 3
        steps.append("✅ Step 3: ChromaDB 벡터 검색")
        vector_results = self._vector_search(query, analysis)
        steps.append(f"   → 벡터 결과 {len(vector_results)}개")

        # Step 4
        steps.append(f"✅ Step 4: RRF Fusion (k={rrf_k or self.rrf_k})")
        fused = self._rrf_fusion(graph_results, vector_results, k=rrf_k)
        # 상위 N 개 선택
        fused = fused[:final_top_k]
        steps.append(f"   → 통합 컨텍스트 {len(fused)}개 (top_k={final_top_k})")

        # Step 5
        steps.append("✅ Step 5: 교차 소스 컨텍스트 구성")
        context = self._build_context(fused)

        # Step 6
        steps.append(f"✅ Step 6: {'LLM' if self._use_llm else '템플릿'} 답변 생성")
        answer = self._generate_answer(query, context, analysis)

        # 출처 목록
        sources = []
        for ctx in fused:
            if ctx.source_url or ctx.source_doc:
                sources.append({
                    "doc": ctx.source_doc,
                    "title": ctx.title,
                    "url": ctx.source_url,
                    "score": round(ctx.rrf_score, 4),
                    "origin": ctx.origin,
                })

        confidence = min(len(fused) / final_top_k, 1.0)

        return SearchResult(
            query=query,
            answer=answer,
            graph_results=[self._graph_item_to_text(r)[:200] for r in graph_results],
            vector_results=[{"content": r.content[:200], "source": r.source_doc,
                             "score": round(r.score, 3)} for r in vector_results],
            fused_contexts=[{"content": c.content[:200], "source": c.source_doc,
                             "score": round(c.rrf_score, 4), "origin": c.origin}
                            for c in fused],
            sources=sources,
            error_codes_found=analysis["error_codes"],
            processing_steps=steps,
            confidence=confidence,
        )

    def search_vector_only(self, query: str, top_k: int = None) -> SearchResult:
        """벡터만 검색"""
        if top_k is None:
            top_k = self.final_top_k
        
        steps = [
            "✅ Step 1: 쿼리 분석 (엔티티·의도 추출)",
            "✅ Step 2: 벡터 검색만 실행 (그래프 탐색 생략)"
        ]
        
        analysis = self._analyze_query(query)
        vector_results = self._vector_search(query, analysis)
        vector_results = vector_results[:top_k]
        
        steps.append(f"   → 벡터 결과 {len(vector_results)}개")
        
        # 벡터 결과를 FusedContext 로 변환
        fused = []
        for item in vector_results:
            fused.append(FusedContext(
                content=item.content,
                source_doc=item.source_doc,
                source_url=item.source_url,
                title=item.metadata.get("title", ""),
                rrf_score=item.score,
                origin="vector",
                metadata=item.metadata,
            ))
        
        steps.append("✅ Step 3: 컨텍스트 구성")
        context = self._build_context(fused)
        
        steps.append(f"✅ Step 4: {'LLM' if self._use_llm else '템플릿'} 답변 생성")
        answer = self._generate_answer(query, context, analysis)
        
        # 출처 목록
        sources = []
        for ctx in fused:
            if ctx.source_url or ctx.source_doc:
                sources.append({
                    "doc": ctx.source_doc,
                    "title": ctx.title,
                    "url": ctx.source_url,
                    "score": round(ctx.rrf_score, 4),
                    "origin": ctx.origin,
                })
        
        confidence = min(len(fused) / top_k, 1.0)
        
        return SearchResult(
            query=query,
            answer=answer,
            graph_results=[],
            vector_results=[{"content": r.content[:200], "source": r.source_doc,
                             "score": round(r.score, 3)} for r in vector_results],
            fused_contexts=[{"content": c.content[:200], "source": c.source_doc,
                             "score": round(c.rrf_score, 4), "origin": c.origin}
                            for c in fused],
            sources=sources,
            error_codes_found=analysis["error_codes"],
            processing_steps=steps,
            confidence=confidence,
        )

    def search_graph_only(self, query: str, top_k: int = None) -> SearchResult:
        """그래프만 검색"""
        if top_k is None:
            top_k = self.final_top_k
        
        steps = [
            "✅ Step 1: 쿼리 분석 (엔티티·의도 추출)",
            "✅ Step 2: 그래프 탐색만 실행 (벡터 검색 생략)"
        ]
        
        analysis = self._analyze_query(query)
        graph_results = self._graph_search(analysis)
        graph_results = graph_results[:top_k]
        
        steps.append(f"   → 그래프 결과 {len(graph_results)}개")
        
        # 그래프 결과를 FusedContext 로 변환
        fused = []
        for item in graph_results:
            content = self._graph_item_to_text(item)
            item_map = item if isinstance(item, dict) else {"_raw": item}
            url = str(item_map.get("feature_url") or item_map.get("url") or "")
            
            fused.append(FusedContext(
                content=content,
                source_doc=str(item_map.get("feature") or item_map.get("code") or ""),
                source_url=url,
                title=str(item_map.get("feature") or item_map.get("title") or ""),
                rrf_score=1.0,  # 그래프만 검색이므로 점수 없음
                origin="graph",
                metadata=item if isinstance(item, dict) else {"raw": item},
            ))
        
        steps.append("✅ Step 3: 컨텍스트 구성")
        context = self._build_context(fused)
        
        steps.append(f"✅ Step 4: {'LLM' if self._use_llm else '템플릿'} 답변 생성")
        answer = self._generate_answer(query, context, analysis)
        
        # 출처 목록
        sources = []
        for ctx in fused:
            if ctx.source_url or ctx.source_doc:
                sources.append({
                    "doc": ctx.source_doc,
                    "title": ctx.title,
                    "url": ctx.source_url,
                    "score": round(ctx.rrf_score, 4),
                    "origin": ctx.origin,
                })
        
        confidence = min(len(fused) / top_k, 1.0)
        
        return SearchResult(
            query=query,
            answer=answer,
            graph_results=[self._graph_item_to_text(r)[:200] for r in graph_results],
            vector_results=[],
            fused_contexts=[{"content": c.content[:200], "source": c.source_doc,
                             "score": round(c.rrf_score, 4), "origin": c.origin}
                            for c in fused],
            sources=sources,
            error_codes_found=analysis["error_codes"],
            processing_steps=steps,
            confidence=confidence,
        )
