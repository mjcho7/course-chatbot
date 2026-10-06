# 강의 추천 RAG 챗봇

`data/courses.json`의 강의만 검색하고 추천하는 Streamlit 대시보드입니다. 추천, 수집 데이터, 질문 기록 페이지를 제공합니다.

## 로컬 실행

Python 3.11과 uv가 필요합니다.

```bash
uv sync
uv run streamlit run app.py
```

프로젝트 루트의 `.env`에 다음 값을 둡니다.

```text
OPENAI_API_KEY=your_api_key
```

## 강의 인덱스 갱신

`courses.json`을 수집·갱신한 뒤 [update_faiss.bat](update_faiss.bat)을 실행하세요. 이 프로그램은 강의 ID를 기준으로 추가·변경·삭제를 비교하고, 새 강의와 바뀐 강의만 다시 임베딩합니다. 데이터 변경이 없으면 OpenAI 임베딩 API를 호출하지 않습니다.

명령줄에서는 아래와 같이 실행할 수 있습니다.

```bash
uv run python refresh_index.py
```

검색 화면은 `data/faiss`의 저장된 인덱스만 읽습니다. 인덱스가 없을 때는 먼저 갱신 프로그램을 실행해야 합니다.

질문과 추천 결과, 사용자의 도움 됨/도움 안 됨 평가는 `logs/questions.jsonl`에 저장됩니다.
