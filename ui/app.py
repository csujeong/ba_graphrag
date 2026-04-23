"""
BA Graph RAG — Streamlit UI
탭 구성:
  🔍 검색       — 하이브리드 검색 메인
  🕸️ 지식그래프  — Neo4j 그래프 시각화
  📊 대시보드    — 시스템 통계
  ⚙️ 데이터 수집 — 크롤링/파싱 트리거
"""
import sys, os
from pathlib import Path
# Ensure project root is on sys.path so top-level packages (e.g. `graph`) import correctly
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BA Graph RAG 지식 검색",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.main { background-color: #F0F4F8; }

/* 헤더 */
.header-box {
    background: linear-gradient(135deg, #1B3A5C 0%, #2563A8 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.header-box h1 { color: white; margin: 0; font-size: 1.8rem; }
.header-box p  { color: #AED6F1; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

/* 검색창 */
.search-container {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 1.2rem;
}

/* 결과 카드 */
.result-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    margin-bottom: 0.8rem;
    border-left: 4px solid #2563A8;
}
.result-card.error { border-left-color: #E74C3C; }
.result-card.both  { border-left-color: #1A7A4A; }

/* 배지 */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-graph  { background: #D4E6F1; color: #1B4F72; }
.badge-vector { background: #D5F5E3; color: #1A5276; }
.badge-both   { background: #FDEBD0; color: #784212; }
.badge-error  { background: #FADBD8; color: #922B21; }

/* 출처 링크 */
.source-chip {
    display: inline-block;
    background: #EBF5FB;
    border: 1px solid #AED6F1;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    margin: 3px;
    color: #1B4F72;
}

/* 단계 표시 */
.step-item {
    font-size: 0.85rem;
    color: #2C3E50;
    padding: 3px 0;
}

/* 통계 카드 */
.stat-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.stat-num  { font-size: 2rem; font-weight: 700; color: #1B3A5C; }
.stat-label { font-size: 0.8rem; color: #7F8C8D; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── 세션 초기화 ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="시스템 초기화 중…")
def init_system():
    from graph.graph_builder import GraphBuilder
    from vectordb.vector_store import VectorStore
    from search.hybrid_search import HybridSearchEngine

    graph = GraphBuilder()
    vector = VectorStore()
    engine = HybridSearchEngine(graph, vector)

    # 벡터 DB 비어있으면 데모 데이터 로드
    stats = vector.get_stats()
    if stats.get("ba_unified", 0) == 0:
        vector.load_demo_data()
        # 그래프 데모 노드 생성
        _build_demo_graph(graph)

    return graph, vector, engine


def _build_demo_graph(graph):
    """데모 그래프 데이터"""
    # Document 노드
    graph.upsert_document("user_manual", "사용자매뉴얼", "user", "pdf",
        "data/raw/Brity Automation 사용자 매뉴얼4.1.pdf")
    # graph.upsert_document("user_manual", "사용자매뉴얼", "user", "html",
    #     "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_User_4_1.html")
    graph.upsert_document("troubleshoot", "장애조치가이드", "troubleshoot", "html",
        "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html")
    graph.upsert_document("admin_manual", "관리자매뉴얼", "admin", "pdf",
        "data/raw/Brity Automation 관리자 매뉴얼4.1.pdf")
    graph.upsert_document("install_manual", "설치매뉴얼", "install", "pdf",
        "data/raw/Brity Automation 설치 매뉴얼4.1.pdf")
    
    # Feature 노드
    for feat in ["스케줄러", "봇 배포", "워크플로우", "모니터링"]:
        graph.upsert_feature(feat, "사용자매뉴얼")

    # 오류 코드
    graph.upsert_error("ERR-4021", "ERR-4021 연결 타임아웃",
        "네트워크 방화벽에 의해 BA Server 포트(8080)가 차단된 경우 발생합니다.",
        "Critical",
        "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html#4d56856d8b67d905",
        "4d56856d8b67d905")
    graph.upsert_error("ERR-2013", "ERR-2013 권한 없음",
        "현재 사용자 계정에 봇 실행 권한이 없습니다.",
        "Warning",
        "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html#err2013")

    # Resolution
    graph.upsert_resolution("res_4021",
        "1) 방화벽에서 TCP 8080 포트 허용\n2) BA Runner 서비스 재시작\n3) 연결 재시도",
        "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html#4d56856d8b67d905")
    graph.upsert_resolution("res_2013",
        "1) 관리자에게 BA_RUNNER 역할 부여 요청\n2) 관리매뉴얼 권한 설정 참조",
        "https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Admin_4_1.html#user_mgmt")

    # Scenario
    graph.upsert_scenario("scen_basic_001", "기본 봇 생성 실습", "기본",
        "BA기본과정 교육교재", "2장 기본 기능 실습", 15)
    graph.upsert_scenario("scen_adv_001", "E2E 구매 자동화 시나리오", "심화",
        "BA심화과정 교육교재", "2장 복잡한 워크플로우 설계", 30)

    # 관계
    graph.add_may_cause("스케줄러", "ERR-4021", "포트 차단 시")
    graph.add_may_cause("봇 배포", "ERR-2013", "권한 미설정 시")
    graph.add_resolved_by("ERR-4021", "res_4021")
    graph.add_resolved_by("ERR-2013", "res_2013")
    graph.add_demonstrated_in("스케줄러", "scen_basic_001")
    graph.add_demonstrated_in("봇 배포", "scen_adv_001")


# ── 사이드바 ─────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### 🤖 기술지원 어시스턴트")
        st.markdown("**Graph RAG 정보검색**")
        st.divider()

        st.markdown("#### 📚 지식 소스")
        sources = {
            "사용자매뉴얼": "🟢",
            "관리매뉴얼": "🟢",
            "설치매뉴얼": "🟢",
            "장애조치가이드": "🟢",
            "기본과정 교육교재": "🟡",
            "E2E 심화 교육교재": "🟡",
        }
        for src, status in sources.items():
            st.markdown(f"{status} {src}")

        st.divider()
        st.markdown("#### ⚙️ 검색 설정")
        rrf_k = st.slider("RRF k 값", 10, 100, 60, 10)
        top_k = st.slider("결과 개수", 3, 10, 5)

        st.divider()
        st.markdown("#### 📊 검색 방식")
        search_mode = st.radio(
            "검색 방식 선택",
            ["hybrid-RRF", "vector-only", "graph-only"],
            help="hybrid-RRF: 그래프 + 벡터 RRF 통합, vector-only: 벡터만 검색, graph-only: 그래프만 검색"
        )

        st.divider()
        st.markdown("""
        <small>
        🔵 <b>Neo4j</b>: 그래프 탐색<br>
        🟣 <b>ChromaDB</b>: 벡터 검색<br>
        🟠 <b>RRF</b>: 하이브리드 통합
        </small>
        """, unsafe_allow_html=True)

        return {"rrf_k": rrf_k, "top_k": top_k, "search_mode": search_mode}


# ══════════════════════════════════════════════════════════════════
# 탭 1: 검색
# ══════════════════════════════════════════════════════════════════
def tab_search(engine, settings):
    st.markdown("""
    <div class="header-box">
        <h1>🔍 Graph RAG 하이브리드 지식 검색</h1>
        <p>Neo4j 지식그래프 + ChromaDB 벡터 검색 | Brity Automation 기술문서 통합 Q&A</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 예시 질문 버튼 ──────────────────────────────────────────
    st.markdown("**💡 예시 질문 (클릭하여 검색)**")
    examples = [
        "스케줄러 설정하는 방법은?",
        "4021 오류코드 해결 방법",
        "봇 배포 절차와 관련 실습",
        "설치 후 초기 관리자 설정",
        "워크플로우 생성 단계별 안내",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()

    # ── 검색 입력 ────────────────────────────────────────────────
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""
    
    query = st.text_area(
        "질문을 입력하세요",
        height=80,
        placeholder="예: 4021 오류코드가 발생했는데 어떻게 해결하나요?",
        key="query_input",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_btn = st.button("🔍 검색", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ 초기화", use_container_width=True):
            st.session_state.pop("query_input", None)
            st.session_state.pop("last_result", None)
            st.rerun()

    if not search_btn and "last_result" not in st.session_state:
        _show_intro()
        return

    if search_btn and query.strip():
        search_mode = settings["search_mode"]
        search_label = {
            "hybrid-RRF": "🔄 하이브리드 검색 (그래프 + 벡터 RRF)",
            "vector-only": "🔄 벡터만 검색",
            "graph-only": "🔄 그래프만 검색"
        }.get(search_mode, "🔄 검색")
        
        with st.spinner(search_label):
            t0 = time.time()
            if search_mode == "hybrid-RRF":
                result = engine.search(query.strip(), rrf_k=settings["rrf_k"], top_k=settings["top_k"])
            elif search_mode == "vector-only":
                result = engine.search_vector_only(query.strip(), top_k=settings["top_k"])
            elif search_mode == "graph-only":
                result = engine.search_graph_only(query.strip(), top_k=settings["top_k"])
            else:
                result = engine.search(query.strip(), rrf_k=settings["rrf_k"], top_k=settings["top_k"])
            elapsed = time.time() - t0
        st.session_state["last_result"] = result
        st.session_state["last_elapsed"] = elapsed
        st.session_state["last_search_mode"] = search_mode

    result = st.session_state.get("last_result")
    if not result:
        return

    elapsed = st.session_state.get("last_elapsed", 0)

    # ── 결과 헤더 ────────────────────────────────────────────────
    search_mode = st.session_state.get("last_search_mode", "hybrid-RRF")
    mode_badge = {
        "hybrid": "🟠 Hybrid-RRF",
        "vector-only": "🟣 Vector-only",
        "graph-only": "🔵 Graph-only"
    }.get(search_mode, "Hybrid-RRF")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("검색 방식", mode_badge)
    c2.metric("응답 시간", f"{elapsed:.2f}s")
    c3.metric("결과 개수", f"{len(result.fused_contexts)}개")
    c4.metric("신뢰도", f"{result.confidence:.0%}")

    # 오류 코드 감지 알림
    if result.error_codes_found:
        st.error(f"🔴 감지된 오류 코드: {', '.join(result.error_codes_found)}")

    st.divider()

    # ── 메인 답변 ────────────────────────────────────────────────
    result_col, info_col = st.columns([3, 1])

    with result_col:
        st.markdown("### 📝 답변")
        st.markdown(
            f'<div class="result-card">{result.answer}</div>',
            unsafe_allow_html=True,
        )

        # 출처
        if result.sources:
            st.markdown("#### 📎 참조 출처")
            for src in result.sources:
                badge_cls = "badge-both" if src["origin"] == "both" else \
                            "badge-graph" if src["origin"] == "graph" else "badge-vector"
                badge_txt = "그래프+벡터" if src["origin"] == "both" else \
                            "그래프" if src["origin"] == "graph" else "벡터"
                url_html = f'<a href="{src["url"]}" target="_blank">🔗 링크</a>' \
                           if src.get("url") else ""
                st.markdown(
                    f'<div class="source-chip">'
                    f'<span class="badge {badge_cls}">{badge_txt}</span>'
                    f' <b>{src["doc"]}</b> — {src["title"][:40]}  {url_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with info_col:
        # 처리 단계
        st.markdown("#### 🔄 처리 단계")
        for step in result.processing_steps:
            st.markdown(f'<div class="step-item">{step}</div>', unsafe_allow_html=True)

    # ── 상세 결과 탭 ─────────────────────────────────────────────
    st.divider()
    detail_tabs = st.tabs(["🕸️ 그래프 결과", "🔢 벡터 결과", "🔀 통합 컨텍스트"])

    with detail_tabs[0]:
        if result.graph_results:
            for i, r in enumerate(result.graph_results, 1):
                st.markdown(f"**{i}.** {r}")
        else:
            st.info("그래프 탐색 결과 없음 — Neo4j 연결 또는 데이터 확인 필요")

    with detail_tabs[1]:
        if result.vector_results:
            for i, r in enumerate(result.vector_results, 1):
                score_bar = "🟩" * int(r["score"] * 10) + "⬜" * (10 - int(r["score"] * 10))
                st.markdown(
                    f"**{i}. [{r['source']}]** `score: {r['score']:.3f}` {score_bar}\n\n"
                    f"{r['content']}"
                )
                st.divider()
        else:
            st.info("벡터 검색 결과 없음 — ChromaDB 데이터 확인 필요")

    with detail_tabs[2]:
        if result.fused_contexts:
            for i, ctx in enumerate(result.fused_contexts, 1):
                origin_badge = {
                    "both": "🟠 그래프+벡터",
                    "graph": "🔵 그래프",
                    "vector": "🟣 벡터",
                }.get(ctx["origin"], ctx["origin"])
                st.markdown(
                    f"**{i}. RRF Score: {ctx['score']:.4f}** | {origin_badge} | {ctx['source']}\n\n"
                    f"{ctx['content']}"
                )
                st.divider()


def _show_intro():
    st.markdown("""
    <div style="background:white; border-radius:12px; padding:2rem; margin-top:1rem;
                box-shadow: 0 2px 12px rgba(0,0,0,0.06);">
        <h3 style="color:#1B3A5C;">📖 시스템 소개</h3>
        <p>이 시스템은 Brity Automation의 기술 문서를 <b>Neo4j 지식그래프</b>와 
        <b>ChromaDB 벡터 DB</b>를 결합한 <b>Graph RAG 하이브리드 검색</b>으로 제공합니다.</p>
        <hr>
        <h4>검색 가능한 문서</h4>
        <ul>
            <li>🌐 <b>사용자매뉴얼 v4.1</b> — 기능 사용법, 메뉴 경로</li>
            <li>🌐 <b>관리매뉴얼 v4.1</b> — 시스템 설정, 사용자 권한</li>
            <li>🌐 <b>설치매뉴얼 v4.1</b> — 설치 절차, 환경 구성</li>
            <li>🌐 <b>장애조치가이드 v4.1</b> — 오류코드, 해결 방법</li>
            <li>📄 <b>기본과정 교육교재</b> — 기본 기능 실습</li>
            <li>📄 <b>E2E 심화 교육교재</b> — End-to-End 구축 실습</li>
        </ul>
        <h4>검색 방식</h4>
        <ol>
            <li>쿼리에서 <b>기능명·오류코드</b> 추출</li>
            <li><b>Neo4j</b>에서 관련 노드 관계 탐색</li>
            <li><b>ChromaDB</b>에서 의미 기반 유사 청크 검색</li>
            <li><b>RRF</b>로 두 결과 통합 순위화</li>
            <li><b>LLM</b>이 출처 명시 답변 생성</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 탭 2: 지식그래프 시각화
# ══════════════════════════════════════════════════════════════════
def tab_graph(graph):
    st.markdown("### Neo4j 지식그래프 시각화")

    # GraphVisualizer 사용
    from graph.visualize_graph import GraphVisualizer

    viz = GraphVisualizer()

    # HTML 생성
    st.markdown("#### Interactive Graph Visualization")
    html_file = viz.generate_html_viz(output_file="ui/graph_visualization.html")

    # HTML 파일 읽어서 표시
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # iframe 으로 표시
    st.components.v1.html(html_content, height=800, scrolling=True)

# ══════════════════════════════════════════════════════════════════
# 탭 3: 대시보드
# ══════════════════════════════════════════════════════════════════
def tab_dashboard(graph, vector):
    st.markdown("### 📊 시스템 대시보드")

    # GraphVisualizer 에서 데이터 가져오기
    from graph.visualize_graph import GraphVisualizer
    
    viz = GraphVisualizer()
    graph_data = viz.export_graph_data(max_nodes=2000)
    
    nodes = graph_data["nodes"]
    relationships = graph_data["relationships"]
    
    # 페이지 노드 필터링 후 통계 계산
    node_counts = {}
    for node in nodes:
        label = node.get("label", "Unknown")
        # label 이 None 또는 null 인 경우 'Page_or_None'로 처리
        if label is None or label == "null" or label == "":
            label = "Page_or_None"
        node_label = (node.get("name") or node.get("title") or node.get("code") or 
                      node.get("question") or f"{label} {node['id']}")
        
        # 페이지 노드 제외
        if str(node_label).startswith("페이지 "):
            continue
        
        node_counts[label] = node_counts.get(label, 0) + 1
    
    # 관계 타입 카운트
    rel_counts = {}
    for rel in relationships:
        rel_type = rel.get("type", "Unknown")
        # rel_type 이 None 또는 null 인 경우 'Page_or_None'로 처리
        if rel_type is None or rel_type == "null" or rel_type == "":
            rel_type = "Page_or_None"
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    # 필터링된 노드 수
    filtered_node_count = sum(node_counts.values())
    filtered_rel_count = len(relationships)

    v_stats = vector.get_stats()

    # ── 전체 현황 ──────────────────────────────────────────────
    st.markdown("#### Neo4j 그래프 노드 현황")
    
    # 상위 10 개 노드 타입만 표시
    sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    node_types = [n[0] for n in sorted_nodes]
    node_counts_list = [n[1] for n in sorted_nodes]

    fig_nodes = px.bar(
        x=node_types, y=node_counts_list,
        color=node_types,
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={"x": "노드 유형 (Page & None nodes not included)", "y": "개수"},
        text=node_counts_list,
    )
    fig_nodes.update_traces(textposition="outside")
    fig_nodes.update_layout(height=320, showlegend=False,
                            plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC")
    st.plotly_chart(fig_nodes, use_container_width=True)

    st.divider()

    # ── 관계 유형 현황 ─────────────────────────────────────────
    st.markdown("#### 그래프 관계 (Edge) 유형")
    
    sorted_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    rel_types_list = [r[0] for r in sorted_rels]
    rel_counts_list = [r[1] for r in sorted_rels]

    fig_rels = px.bar(
        x=rel_types_list, y=rel_counts_list,
        color=rel_types_list,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"x": "관계 유형", "y": "개수"},
        text=rel_counts_list,
    )
    fig_rels.update_traces(textposition="outside")
    fig_rels.update_layout(height=320, showlegend=False,
                           plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC")
    st.plotly_chart(fig_rels, use_container_width=True)

    st.divider()

    # ── ChromaDB 컬렉션 현황 ───────────────────────────────────
    st.markdown("#### ChromaDB 컬렉션 현황")

    col_names = list(v_stats.keys())
    col_counts = list(v_stats.values())

    fig_vecs = px.bar(
        x=col_names, y=col_counts,
        color=col_counts,
        color_continuous_scale="Blues",
        labels={"x": "컬렉션", "y": "청크 수"},
        text=col_counts,
    )
    fig_vecs.update_traces(textposition="outside")
    fig_vecs.update_layout(height=320, showlegend=False,
                            plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
                            coloraxis_showscale=False)
    st.plotly_chart(fig_vecs, use_container_width=True)

    st.divider()

    # ── 소스별 분포 파이차트 ────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 그래프 노드 분포")
        if node_counts_list:
            fig_pie = px.pie(names=node_types, values=node_counts_list,
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### 관계 유형 분포")
        if rel_counts_list:
            fig_rel_pie = px.pie(names=rel_types_list, values=rel_counts_list,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_rel_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_rel_pie, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 상세 통계")
    import pandas as pd
    detail = {
        "항목": ["총 그래프 노드", "총 관계(엣지)", "총 벡터 청크",
                 "Feature 노드", "ErrorCode 노드", "통합 컬렉션 청크"],
        "수치": [
        ],
    }
    # 사용 안함 - 아래 새 형식 사용
    # st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)
    
    # 요약 통계 카드
    stat_cols = st.columns(4)
    stat_cols[0].markdown(f'<div class="stat-card"><div class="stat-num">{filtered_node_count}</div><div class="stat-label">총 노드 (필터링)</div></div>', unsafe_allow_html=True)
    stat_cols[1].markdown(f'<div class="stat-card"><div class="stat-num">{filtered_rel_count}</div><div class="stat-label">총 관계</div></div>', unsafe_allow_html=True)
    stat_cols[2].markdown(f'<div class="stat-card"><div class="stat-num">{len(node_counts)}</div><div class="stat-label">노드 유형</div></div>', unsafe_allow_html=True)
    stat_cols[3].markdown(f'<div class="stat-card"><div class="stat-num">{len(rel_counts)}</div><div class="stat-label">관계 유형</div></div>', unsafe_allow_html=True)

    st.divider()
    
    # 상세 통계 테이블
    node_df = pd.DataFrame([{"유형": nt, "개수": cnt} for nt, cnt in sorted_nodes])
    st.dataframe(node_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# 탭 4: 데이터 수집
# ══════════════════════════════════════════════════════════════════
def tab_ingest(graph, vector):
    st.markdown("### ⚙️ 데이터 수집 및 인덱싱")

    st.info("""
    **수집 순서**: HTML크롤링/PDF문서 → Neo4j 그래프 구축 → ChromaDB 벡터 인덱싱
    
    ⚠️ PDF 매뉴얼 및 교육교재는 `/data/raw/` 폴더에 파일을 먼저 복사한 후 실행하세요.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌐 HTML 매뉴얼 수집")
        st.markdown("""
        | 소스 | URL |
        |------|-----|
        | 사용자매뉴얼 | [링크](https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_User_4_1.html) |
        | 관리매뉴얼 | [링크](https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Admin_4_1.html) |
        | 설치매뉴얼 | [링크](https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html) |
        | 장애조치가이드 | [링크](https://support.samsungsds.com/BA/BrityAutomation_KOR/help@BrityAutomation_Install_4_1.html#4d56856d8b67d905) |
        """)

        # 선택적 HTML 크롤링: 문서별 체크박스 표시 후 선택된 문서만 크롤링
        from crawler.html_crawler import BAManualCrawler
        st.markdown("#### 처리할 HTML 매뉴얼 선택")
        html_sources = BAManualCrawler.SOURCES
        html_selected = {}
        cols_check = st.columns(2)
        i = 0
        for key, info in html_sources.items():
            col = cols_check[i % 2]
            html_selected[key] = col.checkbox(info["name"], value=True, key=f"html_{key}")
            i += 1

        if st.button("🕷️ HTML 매뉴얼 크롤링 시작"):
            selected_keys = [k for k, v in html_selected.items() if v]
            st.write(f"선택된 키: {selected_keys}")  # 디버깅용
            if not selected_keys:
                st.warning("크롤할 문서를 하나 이상 선택하세요.")
            else:
                with st.spinner("크롤링 진행 중… (수 분 소요)"):
                    try:
                        crawler = BAManualCrawler(headless=True)
                        crawler._init_driver()
                        total = 0
                        prog = st.progress(0)
                        try:
                            for idx, key in enumerate(selected_keys):
                                sections = crawler.crawl_one(key)
                                st.write(f"{key}: {len(sections or [])} 섹션 수집됨")  # 디버깅용
                                total += len(sections or [])
                                # 그래프/벡터에 추가
                                if sections:
                                    graph.build_from_html_sections(sections, key)
                                    for sec in sections:
                                        vector.add_html_section(sec, key)
                                prog.progress((idx + 1) / len(selected_keys))
                        finally:
                            if crawler.driver:
                                crawler.driver.quit()

                        st.success(f"✅ HTML 수집 완료! 총 {total}개 섹션")
                    except Exception as e:
                        st.error(f"크롤링 오류: {e}")

    with col2:
        st.markdown("#### 📄 PDF 매뉴얼 및 교육교재 수집")
        basic_file = "data/raw/BA기본과정_기본기능 및 실습 교육교재.pdf"
        adv_file = "data/raw/BA심화과정_E2E구축방법교육교재.pdf"
        install_file = "data/raw/Brity Automation 설치 매뉴얼4.1.pdf"
        admin_file = "data/raw/Brity Automation 관리자 매뉴얼4.1.pdf"
        user_file = "data/raw/Brity Automation 사용자 매뉴얼4.1.pdf"

        basic_exists = os.path.exists(basic_file)
        adv_exists = os.path.exists(adv_file)
        install_exists = os.path.exists(install_file)
        admin_exists = os.path.exists(admin_file)
        user_exists = os.path.exists(user_file)

        st.markdown(f"- 기본과정: {'✅ 파일 있음' if basic_exists else '⚠️ 파일 없음'}")
        st.markdown(f"- 심화과정: {'✅ 파일 있음' if adv_exists else '⚠️ 파일 없음'}")
        st.markdown(f"- 설치매뉴얼: {'✅ 파일 있음' if install_exists else '⚠️ 파일 없음'}")
        st.markdown(f"- 관리매뉴얼: {'✅ 파일 있음' if admin_exists else '⚠️ 파일 없음'}")
        st.markdown(f"- 사용자매뉴얼: {'✅ 파일 있음' if user_exists else '⚠️ 파일 없음'}")

        # 파일 업로드 섹션 - 파일이 없는 경우에만 표시
        missing_files = []
        if not basic_exists: missing_files.append("기본과정 교육교재")
        if not adv_exists: missing_files.append("심화과정 교육교재")
        if not install_exists: missing_files.append("설치매뉴얼")
        if not admin_exists: missing_files.append("관리매뉴얼")
        if not user_exists: missing_files.append("사용자매뉴얼")

        if missing_files:
            st.markdown("##### 📤 파일 업로드")
            st.info(f"업로드할 파일: {', '.join(missing_files)}")
            
            uploaded_file = st.file_uploader(
                "PDF 파일을 선택하여 업로드하세요",
                type=["pdf"],
                key="pdf_uploader_main"
            )
            
            if uploaded_file is not None:
                uploaded_filename = uploaded_file.name.lower()
                target_path = None
                
                if "기본과정" in uploaded_filename or "기본기능" in uploaded_filename:
                    target_path = basic_file
                elif "심화과정" in uploaded_filename or "e2e" in uploaded_filename:
                    target_path = adv_file
                elif "설치" in uploaded_filename:
                    target_path = install_file
                elif "관리자" in uploaded_filename:
                    target_path = admin_file
                elif "사용자" in uploaded_filename:
                    target_path = user_file
                
                if target_path:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with open(target_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ '{uploaded_file.name}' 파일이 '{target_path}'에 저장되었습니다!")
                    st.rerun()
                else:
                    st.warning("⚠️ 해당 PDF 파일은 시스템에서 인식하지 못했습니다. 파일명을 확인해주세요.")

        st.markdown("---")

        # 문서별 체크박스 표시 — 체크된 문서만 파싱/인덱싱
        st.markdown("#### 처리할 PDF/교재 선택")
        pdf_checks_col1, pdf_checks_col2 = st.columns(2)
        proc_basic = pdf_checks_col1.checkbox("기본과정 교육교재", value=basic_exists, key="pdf_basic")
        proc_adv = pdf_checks_col1.checkbox("심화과정 교육교재", value=adv_exists, key="pdf_adv")
        proc_install = pdf_checks_col2.checkbox("설치매뉴얼", value=install_exists, key="pdf_install")
        proc_admin = pdf_checks_col2.checkbox("관리매뉴얼", value=admin_exists, key="pdf_admin")
        proc_user = pdf_checks_col2.checkbox("사용자매뉴얼", value=user_exists, key="pdf_user")

        if st.button("📄 PDF 매뉴얼 및 교육교재 처리 시작", type="primary"):
            selected = []
            if proc_basic: selected.append((basic_file, "training_basic"))
            if proc_adv: selected.append((adv_file, "training_advanced"))
            if proc_install: selected.append((install_file, "install_manual"))
            if proc_admin: selected.append((admin_file, "admin_manual"))
            if proc_user: selected.append((user_file, "user_manual"))

            if not selected:
                st.warning("처리할 PDF를 하나 이상 선택하세요.")
            else:
                with st.spinner("PDF 파싱 및 인덱싱 중…"):
                    try:
                        from parser.pdf_parser import PDFParser
                        parser = PDFParser()
                        prog = st.progress(0)
                        for i, (filepath, col) in enumerate(selected):
                            chunks = parser.parse(filepath, col)
                            graph.build_from_pdf_chunks(chunks, col)
                            for c in chunks:
                                vector.add_pdf_chunk(c, col)
                            prog.progress((i + 1) / len(selected))

                        st.success(f"✅ 선택한 PDF 문서 처리 완료! 처리한 문서 수: {len(selected)}")
                    except Exception as e:
                        st.error(f"PDF 처리 오류: {e}")

    st.divider()

    # 데모 데이터 로드
    st.markdown("#### 🧪 데모 데이터")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("🔄 데모 데이터 (재)로드"):
            with st.spinner("데모 데이터 로드 중…"):
                vector.load_demo_data()
                _build_demo_graph(graph)
                st.success("✅ 데모 데이터 로드 완료!")
                st.cache_resource.clear()
                st.rerun()
    with col_d2:
        if st.button("🗑️ 전체 그래프 초기화", type="secondary"):
            if st.session_state.get("confirm_clear"):
                graph.clear_all()
                st.success("Neo4j 그래프 초기화 완료")
                st.session_state["confirm_clear"] = False
            else:
                st.session_state["confirm_clear"] = True
                st.warning("한 번 더 클릭하면 모든 그래프 데이터가 삭제됩니다!")


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main():
    # 시스템 초기화
    graph, vector, engine = init_system()
    settings = sidebar()

    # 탭 구성
    tabs = st.tabs(["🔍 검색", "🕸️ 지식그래프", "📊 대시보드", "⚙️ 데이터 수집"])

    with tabs[0]:
        tab_search(engine, settings)
    with tabs[1]:
        tab_graph(graph)
    with tabs[2]:
        tab_dashboard(graph, vector)
    with tabs[3]:
        tab_ingest(graph, vector)


if __name__ == "__main__":
    main()
