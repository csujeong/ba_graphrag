# BA Graph RAG — Brity Automation 지식 검색 시스템

## 📦 프로젝트 구조

```
ba_graphrag/
├── crawler/
│   └── html_crawler.py      # Selenium + BS4 HTML 매뉴얼 크롤러
├── parser/
│   └── pdf_parser.py        # PyMuPDF + pdfplumber PDF 파서
├── graph/
│   └── graph_builder.py     # Neo4j 지식그래프 빌더
├── vectordb/
│   └── vector_store.py      # ChromaDB 벡터 스토어 (7개 컬렉션)
├── search/
│   └── hybrid_search.py     # RRF 하이브리드 검색 엔진
├── api/
│   └── server.py            # FastAPI 백엔드
├── ui/
│   └── app.py               # Streamlit UI (4개 탭)
├── config/
│   └── ontology.yaml        # 도메인 온톨로지 설정
├── utils/
│   └── common.py            # 공통 유틸리티
├── data/
│   ├── raw/                 # PDF 교육교재 원본 저장 위치
│   └── processed/           # 처리된 청크 JSON
├── docker-compose.yml       # Neo4j + ChromaDB + API + UI
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
└── .env.example
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 등 설정
```

### 2. Docker Compose로 전체 실행
```bash
docker-compose up -d
# UI: http://localhost:8501
# API: http://localhost:8080
# Neo4j Browser: http://localhost:7474
```

### 3. 로컬 실행 (개발용)
```bash
pip install -r requirements.txt

# Neo4j, ChromaDB는 별도 실행 필요 (또는 Mock 모드 자동 사용)

# UI 실행
cd ba_graphrag
streamlit run ui/app.py

# API 서버 실행 (별도 터미널)
python api/server.py
```

### 4. PDF 교육교재 추가
```bash
# data/raw/ 폴더에 PDF 파일 복사
cp "BA기본과정_기본기능 및 실습 교육교재.pdf" data/raw/
cp "BA심화과정_E2E구축방법교육교재.pdf" data/raw/

# UI의 [⚙️ 데이터 수집] 탭에서 PDF 처리 실행
```

## 🔧 주요 기술 스택

| 구성 요소 | 기술 | 역할 |
|-----------|------|------|
| 그래프 DB | Neo4j 5.x + APOC | 지식그래프 저장 및 탐색 |
| 벡터 DB | ChromaDB | 청크 임베딩 및 유사도 검색 |
| 임베딩 | BAAI/BGE-M3 | 한국어+영어 문서 임베딩 |
| LLM | GPT-4o / sLLM | 답변 생성 |
| UI | Streamlit | 4탭 인터페이스 |
| API | FastAPI | REST 백엔드 |

## 📊 검색 파이프라인

```
사용자 질의
    │
    ▼
[Step 1] Query Analyzer (엔티티·의도 추출)
    │
    ├──▶ [Step 2] Neo4j Graph Traversal
    │         Feature → MAY_CAUSE → ErrorCode
    │         ErrorCode → RESOLVED_BY → Resolution
    │
    ├──▶ [Step 3] ChromaDB Vector Search
    │         ba_unified 컬렉션 Top-15
    │
    ▼
[Step 4] RRF Fusion (k=60)
    │
    ▼
[Step 5] Cross-Source Context Builder
    │
    ▼
[Step 6] LLM Answer Generation (출처 인용)
    │
    ▼
[Step 7] Response Formatter
```

## ⚠️ Neo4j / ChromaDB 없이도 동작

Neo4j 또는 ChromaDB가 설치되지 않은 환경에서는 자동으로 **Mock 모드**로 동작합니다.
데모 데이터가 자동 로드되어 UI 시연이 가능합니다.
