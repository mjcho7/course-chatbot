"""강의 데이터를 FAISS에 저장하고 답변을 생성하는 RAG 모듈."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


ROOT = Path(__file__).resolve().parent
COURSES_PATH = ROOT / "data" / "courses.json"
FAISS_PATH = ROOT / "data" / "faiss"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
RETRIEVAL_CANDIDATES = 20
PROMPT_VERSION = "related-course-v3"
NO_RECOMMENDATION = "추천할 만한 강의를 찾지 못했습니다."
MIN_RELATED_SIMILARITY = 0.35

# 한글 질문과 데이터의 영문 키워드를 함께 찾기 위한 대표 표기다.
KEYWORD_ALIASES = {
    "엑셀": {"엑셀", "excel", "spreadsheet", "스프레드시트"},
}
STOP_WORDS = {"강의", "추천", "배우고", "싶어요", "알려줘", "방법", "관련"}


def _require_api_key() -> None:
    """프로젝트 루트의 .env에서 API 키를 읽고 존재 여부를 확인한다."""
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("프로젝트 루트의 .env에 OPENAI_API_KEY를 설정해 주세요.")


def _to_document(course: dict[str, Any]) -> Document:
    """강의 하나를 하나의 문서로 변환한다."""
    keywords = ", ".join(course.get("keywords") or [])
    content = "\n".join(
        [
            f"제목: {course.get('title', '')}",
            f"소개: {course.get('description', '')}",
            f"키워드: {keywords}",
        ]
    )
    metadata = {
        "course_id": str(course["id"]),
        "title": course.get("title", ""),
        "level": course.get("level", ""),
        "hours": course.get("hours") or "정보 없음",
        "category": course.get("category", ""),
        "url": course.get("url", ""),
    }
    return Document(id=str(course["id"]), page_content=content, metadata=metadata)


def _load_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    """이 앱이 생성한 로컬 FAISS 인덱스를 불러온다."""
    return FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def _build_vectorstore(documents: list[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """courses.json 전체를 최초 한 번만 임베딩해 저장한다."""
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=[document.id for document in documents],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_PATH))
    return vectorstore


def _load_documents() -> list[Document]:
    """원본 강의 데이터를 LangChain 문서 목록으로 읽는다."""
    courses = json.loads(COURSES_PATH.read_text(encoding="utf-8"))
    return [_to_document(course) for course in courses]


def get_course_data_summary() -> str:
    """원본 데이터의 최신 수집일과 전체 강의 수를 화면용 문구로 만든다."""
    courses = json.loads(COURSES_PATH.read_text(encoding="utf-8"))
    latest_crawled_at = max(datetime.fromisoformat(course["crawled_at"]) for course in courses)
    return f"{latest_crawled_at.year}년 {latest_crawled_at.month}월 {latest_crawled_at.day}일 수집 자료 기준 · 강의 {len(courses)}개"


def load_vectorstore() -> FAISS:
    """검색용으로 저장된 FAISS 인덱스만 읽는다."""
    _require_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index_file = FAISS_PATH / "index.faiss"
    store_file = FAISS_PATH / "index.pkl"
    if not index_file.exists() or not store_file.exists():
        raise RuntimeError("FAISS 인덱스가 없습니다. update_faiss.bat를 먼저 실행해 주세요.")
    # 이 파일은 이 앱이 data/faiss에 저장한 로컬 인덱스만 읽는다.
    return _load_vectorstore(embeddings)


def sync_vectorstore() -> dict[str, int]:
    """강의 ID를 기준으로 FAISS 인덱스를 증분 갱신한다."""
    _require_api_key()
    documents = _load_documents()
    documents_by_id = {document.id: document for document in documents}
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    index_file = FAISS_PATH / "index.faiss"
    store_file = FAISS_PATH / "index.pkl"

    if not index_file.exists() or not store_file.exists():
        _build_vectorstore(documents, embeddings)
        return {"added": len(documents), "updated": 0, "deleted": 0, "unchanged": 0}

    vectorstore = _load_vectorstore(embeddings)
    existing_ids = set(vectorstore.index_to_docstore_id.values())
    added = updated = deleted = 0

    # 새 강의와 내용 또는 메타데이터가 바뀐 강의만 임베딩한다.
    for document_id, document in documents_by_id.items():
        if document_id not in existing_ids:
            vectorstore.add_documents([document], ids=[document_id])
            added += 1
            continue
        existing_document = vectorstore.docstore.search(document_id)
        if (
            existing_document.page_content != document.page_content
            or existing_document.metadata != document.metadata
        ):
            vectorstore.delete(ids=[document_id])
            vectorstore.add_documents([document], ids=[document_id])
            updated += 1

    # 원본에서 없어진 강의는 검색 결과에도 남지 않도록 제거한다.
    removed_ids = sorted(existing_ids - set(documents_by_id))
    if removed_ids:
        vectorstore.delete(ids=removed_ids)
        deleted = len(removed_ids)

    if added or updated or deleted:
        vectorstore.save_local(str(FAISS_PATH))
    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "unchanged": len(documents) - added - updated,
    }


def retrieve_courses(
    question: str, vectorstore: FAISS, allowed_course_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    """의미 유사도와 핵심 키워드를 함께 고려해 강의 5개를 반환한다."""
    search_count = len(vectorstore.index_to_docstore_id) if allowed_course_ids else RETRIEVAL_CANDIDATES
    results = vectorstore.similarity_search_with_score(question, k=search_count)
    question_terms = {
        term.casefold()
        for term in re.findall(r"[가-힣A-Za-z0-9]+", question)
        if len(term) >= 2 and term.casefold() not in STOP_WORDS
    }
    ranked = [
        {
            "document": document,
            "metadata": document.metadata,
            # 정규화한 내적은 코사인 유사도와 같다.
            "similarity": float(score),
            "keyword_bonus": _keyword_bonus(question_terms, document.page_content),
        }
        for document, score in results
        if allowed_course_ids is None or document.metadata["course_id"] in allowed_course_ids
    ]
    # 화면에는 실제 의미 유사도를 표시하고, 정렬에만 키워드 가점을 사용한다.
    ranked.sort(key=lambda item: item["similarity"] + item["keyword_bonus"], reverse=True)
    return ranked[:TOP_K]


def course_ids_matching_metadata(
    vectorstore: FAISS, levels: set[str] | None = None, max_hours: float | None = None
) -> set[str]:
    """FAISS 문서의 메타데이터로 수강 대상·수강 시간을 걸러 강의 ID를 반환한다."""
    matching_ids: set[str] = set()
    for document_id in vectorstore.index_to_docstore_id.values():
        document = vectorstore.docstore.search(document_id)
        metadata = document.metadata
        if levels and metadata.get("level") not in levels:
            continue
        hours_match = re.search(r"(\d+(?:\.\d+)?)", str(metadata.get("hours", "")))
        hours = float(hours_match.group(1)) if hours_match else None
        if max_hours is not None and (hours is None or hours > max_hours):
            continue
        matching_ids.add(str(metadata["course_id"]))
    return matching_ids


def _keyword_bonus(question_terms: set[str], content: str) -> float:
    """질문 핵심어가 강의 본문에 있으면 검색 순위에 작은 가점을 준다."""
    searchable = content.casefold()
    bonus = 0.0
    for term in question_terms:
        aliases = KEYWORD_ALIASES.get(term, {term})
        if any(alias in searchable for alias in aliases):
            # 대표 표기의 동의어 일치는 일반 단어 일치보다 강하게 반영한다.
            bonus += 0.18 if term in KEYWORD_ALIASES else 0.04
    return bonus


def recommend_courses(
    question: str, vectorstore: FAISS, allowed_course_ids: set[str] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """검색 결과만 근거로 답변 모델이 강의를 추천하게 한다."""
    matches = retrieve_courses(question, vectorstore, allowed_course_ids)
    if not matches:
        return NO_RECOMMENDATION, []
    context = "\n\n".join(
        f"[후보 {rank}]\n{match['document'].page_content}\n"
        f"수강 대상: {match['metadata']['level']}\n"
        f"수강 시간: {match['metadata']['hours']}\n"
        f"분야: {match['metadata']['category']}\n"
        f"주소: {match['metadata']['url']}"
        for rank, match in enumerate(matches, start=1)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 실용적인 강의 추천 도우미입니다. 반드시 제공된 후보 강의 안에서만 답하세요. "
                "후보에 없는 강의, URL, 세부 정보를 만들지 마세요. 완전히 일치하는 강의가 없어도 "
                "주제, 목적, 실무 상황이 가까워 사용자 질문 해결에 도움이 될 후보라면 적극 추천하세요. "
                "후보 목록은 이미 질문과의 의미 유사도로 검색된 결과입니다. 질문의 핵심 주제나 목적을 공유하는 "
                "후보가 하나라도 있으면 '없음'이라고 답하지 말고, 가장 가까운 후보를 추천하세요. "
                "이때 '직접 특화 강의는 아니지만'처럼 관련 범위나 한계를 한 문장으로 밝혀 주세요. "
                "제공된 후보가 질문의 주제와 목적 모두 전혀 무관할 때만 '추천할 만한 강의를 찾지 못했습니다.'라고 답하세요. "
                "추천할 때는 최대 3개만 고르고 제목, 추천 이유, 수강 대상, 수강 시간을 한국어로 간결하게 쓰세요.",
            ),
            ("human", "질문: {question}\n\n후보 강의:\n{context}"),
        ]
    )
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = (prompt | model).invoke({"question": question, "context": context})
    answer = str(response.content).strip()

    # 모델이 지나치게 보수적으로 답해도, 충분히 가까운 검색 결과는 사용자에게 보여 준다.
    if answer == NO_RECOMMENDATION and matches[0]["similarity"] >= MIN_RELATED_SIMILARITY:
        top = matches[0]["metadata"]
        answer = (
            f"가장 관련 있는 강의는 **{top['title']}**입니다.\n\n"
            "질문과 완전히 같은 표현을 쓰지는 않지만, 검색된 후보 중 주제와 목적이 가장 가깝습니다.\n\n"
            f"- **수강 대상:** {top['level']}\n"
            f"- **수강 시간:** {top['hours']}"
        )
    return answer, matches
