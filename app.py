"""강의 추천, 수집 데이터, 질문 기록을 제공하는 Streamlit 대시보드."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from question_log import load_questions, save_question, update_rating
from rag import (
    COURSES_PATH,
    course_ids_matching_metadata,
    get_course_data_summary,
    load_vectorstore,
    recommend_courses,
)


st.set_page_config(page_title="강의 추천 챗봇", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    """대시보드에 맞는 여백, 카드, 색상 스타일을 적용한다."""
    st.markdown(
        """<style>
        .block-container { max-width: 1460px; padding-top: 2.4rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #f4f6fa; }
        [data-testid="stSidebar"] .stButton>button { background: white; border: 1px solid #d9dee9; }
        .hero { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.25rem; }
        .hero h1 { margin:0; font-size:2.7rem; letter-spacing:-0.05em; }
        .muted { color:#6b7280; font-size:.9rem; }.data-note { text-align:right; color:#64748b; font-size:.88rem; }
        .answer-box { background:#fff7ed; border-left:5px solid #fb923c; border-radius:10px; padding:1rem 1.2rem; }
        .course-title { font-size:1.08rem; font-weight:750; line-height:1.45; min-height:3.1rem; }
        .tag { display:inline-block; margin:0 .3rem .45rem 0; padding:.18rem .48rem; border-radius:5px; background:#edf7f0; color:#21834b; font-size:.78rem; }
        .tag.blue { background:#e9f2ff; color:#2666a8; }.tag.gray { background:#f1f5f9; color:#64748b; }
        .badge { display:inline-block; margin-bottom:.65rem; padding:.24rem .5rem; border-radius:5px; background:#fff1f2; color:#ef4444; font-size:.78rem; font-weight:700; }
        .badge.secondary { background:#f1f5f9; color:#64748b; }.course-desc { min-height:4.5rem; color:#334155; line-height:1.6; }
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_courses() -> list[dict]:
    """원본 강의 데이터를 한 번 읽어 화면용으로 재사용한다."""
    return json.loads(Path(COURSES_PATH).read_text(encoding="utf-8"))


@st.cache_resource
def load_store():
    """저장된 FAISS 인덱스를 불러온다."""
    return load_vectorstore()


def hours_number(value: str) -> float | None:
    """'약 10시간' 같은 값을 필터 비교용 숫자로 바꾼다."""
    found = re.search(r"(\d+(?:\.\d+)?)", value or "")
    return float(found.group(1)) if found else None


def render_header() -> None:
    """모든 페이지의 제목과 최신 수집 기준을 표시한다."""
    st.markdown(
        f"""<div class="hero"><div><h1>🎓 강의 추천 챗봇</h1><div class="muted">패스트캠퍼스 공개 강의 정보 기반 · 비공식</div></div>
        <div class="data-note">🗓️ {get_course_data_summary()}</div></div>""",
        unsafe_allow_html=True,
    )


def course_filters(courses: list[dict]) -> tuple[list[dict], dict]:
    """사이드바 조건을 실제 카드와 FAISS 검색 대상에 적용한다."""
    categories = sorted({course["category"] for course in courses})
    subcategories = sorted({course.get("subcategory") for course in courses if course.get("subcategory")})
    levels = sorted({course["level"] for course in courses if course.get("level")})
    # 시간 구간은 실제 강의가 하나 이상 있는 선택지로만 구성한다.
    duration_limits = {"5시간 이하": 5, "10시간 이하": 10, "20시간 이하": 20}
    duration_options = ["전체"] + [
        label
        for label, limit in duration_limits.items()
        if any((hours := hours_number(course.get("hours", ""))) is not None and hours <= limit for course in courses)
    ]
    with st.sidebar:
        st.markdown("### 조건 필터")
        selected_categories = st.multiselect("분야", categories, placeholder="전체")
        selected_subcategories = st.multiselect("세부 분류", subcategories, placeholder="전체")
        selected_levels = st.multiselect("수강 대상", levels, placeholder="전체")
        duration = st.selectbox("수강 시간", duration_options)
        operator_mode = st.toggle("운영자 보기", value=False)
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.turns = []
            st.rerun()
    filtered = []
    for course in courses:
        hours = hours_number(course.get("hours", ""))
        if selected_categories and course["category"] not in selected_categories:
            continue
        if selected_subcategories and course.get("subcategory") not in selected_subcategories:
            continue
        if selected_levels and course["level"] not in selected_levels:
            continue
        max_hours = duration_limits.get(duration)
        if max_hours is not None and (hours is None or hours > max_hours):
            continue
        filtered.append(course)
    return filtered, {
        "filters": {
            "분야": selected_categories or ["전체"],
            "세부 분류": selected_subcategories or ["전체"],
            "수강 대상": selected_levels or ["전체"],
            "수강 시간": duration,
        },
        "metadata_filters": {"levels": set(selected_levels), "max_hours": duration_limits.get(duration)},
        "operator_mode": operator_mode,
    }


def render_course_card(match: dict, label: str, card_key: str, secondary: bool = False) -> None:
    """courses.json 원본 값으로 추천 강의 카드를 그린다."""
    course_id = str(match["metadata"]["course_id"])
    original = next(
        (item for item in st.session_state.courses if str(item["id"]) == course_id), None
    )
    if original is None:
        # 인덱스와 원본 데이터의 동기화가 필요한 경우에는 카드를 표시하지 않는다.
        return
    with st.container(border=True):
        st.markdown(f'<span class="badge {"secondary" if secondary else ""}">{label}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="course-title">{original["title"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="tag">수강 대상 · {original.get("level") or "정보 없음"}</span>'
            f'<span class="tag blue">수강 시간 · {original.get("hours") or "정보 없음"}</span>',
            unsafe_allow_html=True,
        )
        st.link_button(
            "강의 페이지로 가기",
            original["url"],
            key=f"course-{card_key}",
            use_container_width=True,
            type="secondary" if secondary else "primary",
        )


def render_recommendation_page(courses: list[dict], state: dict) -> None:
    """질문 입력, 추천 카드, 평가 버튼을 제공한다."""
    render_header()
    question = st.session_state.pop("quick_question", None) or st.chat_input("무엇을 배우고 싶으세요?")
    if question:
        # 수강 대상·시간은 FAISS 문서 메타데이터에서 먼저 제한한다.
        metadata_ids = course_ids_matching_metadata(
            st.session_state.vectorstore, **state["metadata_filters"]
        )
        allowed_ids = {str(course["id"]) for course in courses} & metadata_ids
        if not allowed_ids:
            st.warning("현재 조건에 맞는 강의가 없습니다. 수강 대상 또는 수강 시간을 넓혀 주세요.")
            return
        with st.spinner("질문에 맞는 강의를 찾고 있습니다."):
            answer, matches = recommend_courses(question, st.session_state.vectorstore, allowed_ids)
        log_id = save_question(question, state["filters"], matches)
        # 빠른 질문 버튼을 누른 뒤 rerun되어도 답변과 카드를 그대로 다시 그린다.
        st.session_state.turns.append(
            {"question": question, "answer": answer, "log_id": log_id, "matches": matches}
        )

    if not st.session_state.turns:
        return

    # 각 질문의 답변·카드를 순서대로 렌더링한다.
    # 따라서 '이어서 좁혀 보기'의 새 결과는 직전 카드 바로 아래에 붙는다.
    for turn_index, current_turn in enumerate(st.session_state.turns):
        answer = current_turn["answer"]
        matches = current_turn.get("matches", [])
        log_id = current_turn["log_id"]
        st.markdown(
            f"<div class='answer-box'><b>{current_turn['question']}</b><br><br>{answer}</div>",
            unsafe_allow_html=True,
        )
        if matches:
            st.markdown("### 추천한 강의")
            left, right = st.columns(2, gap="large")
            with left:
                render_course_card(matches[0], "가장 적합", f"{turn_index}-primary")
            with right:
                render_course_card(matches[1] if len(matches) > 1 else matches[0], "입문용", f"{turn_index}-secondary", secondary=True)

        # 빠른 질문과 평가는 가장 최근 결과에만 표시한다.
        if turn_index != len(st.session_state.turns) - 1:
            st.divider()
            continue
        if not matches:
            return

        st.markdown("#### 이어서 좁혀 보기")
        for column, prompt in zip(st.columns(3), ["더 쉬운 강의만 추천해줘", "10시간 이하 강의만 추천해줘", "다른 강의 더 보여줘"]):
            with column:
                if st.button(prompt, use_container_width=True):
                    st.session_state.quick_question = f"{current_turn['question']}. {prompt}"
                    st.rerun()
        rating_left, rating_right, _ = st.columns([1, 1, 6])
        with rating_left:
            if st.button("👍 도움 됨", key=f"good-{log_id}"):
                update_rating(log_id, "도움 됨")
                st.success("평가를 저장했습니다.")
        with rating_right:
            if st.button("👎 도움 안 됨", key=f"bad-{log_id}"):
                update_rating(log_id, "도움 안 됨")
                st.info("평가를 저장했습니다.")
        if state["operator_mode"]:
            st.caption("운영자 정보: " + " · ".join(f"{item['metadata']['title']} ({item['similarity']:.3f})" for item in matches))


def render_data_page(courses: list[dict]) -> None:
    """수집 데이터의 통계, 목록, 상세 정보를 표시한다."""
    render_header()
    descriptions = [len(course.get("text", course.get("description", ""))) for course in courses]
    metrics = st.columns(4)
    metrics[0].metric("강의 수", f"{len(courses):,}개")
    metrics[1].metric("분야 수", f"{len({course['category'] for course in courses})}개")
    metrics[2].metric("평균 본문 길이", f"{sum(descriptions) // len(descriptions):,}자")
    metrics[3].metric("전체 청크 수", f"{len(courses):,}개")
    categories = sorted({course["category"] for course in courses})
    left, right = st.columns([1, 2])
    selected_categories = left.multiselect("분야", categories, placeholder="전체")
    keyword = right.text_input("제목·본문에서 찾기", placeholder="예: 엑셀, 보고서, Claude")
    visible = [course for course in courses if (not selected_categories or course["category"] in selected_categories) and (not keyword or keyword.casefold() in f"{course['title']} {course.get('description', '')}".casefold())]
    list_tab, detail_tab, stats_tab = st.tabs([f"목록 ({len(visible)})", "강의 상세", "분야별 통계"])
    with list_tab:
        rows = [{"제목": c["title"], "분야": c["category"], "세부 분류": c.get("subcategory", ""), "형태": c.get("format", ""), "수강 대상": c["level"], "수강 시간": c.get("hours") or "정보 없음", "본문 길이": len(c.get("text", c.get("description", ""))), "청크 수": 1, "한줄소개": c.get("description", "")[:90], "링크": c["url"]} for c in visible]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500, hide_index=True, column_config={"링크": st.column_config.LinkColumn("링크", display_text="열기")})
    with detail_tab:
        if visible:
            selected = st.selectbox("강의 선택", visible, format_func=lambda course: course["title"])
            st.subheader(selected["title"])
            st.write(selected.get("description", ""))
            st.link_button("강의 페이지 열기", selected["url"])
    with stats_tab:
        st.bar_chart(pd.DataFrame(visible).groupby("category").size().sort_values(ascending=False))


def render_question_page() -> None:
    """JSONL로 저장된 질문과 도움 평가를 조회한다."""
    render_header()
    questions = load_questions()
    metrics = st.columns(4)
    metrics[0].metric("질문 수", f"{len(questions)}건")
    metrics[1].metric("강의를 못 찾은 질문", f"{sum(item.get('searched_count', 0) == 0 for item in questions)}건")
    metrics[2].metric("도움 안 됨 👎", f"{sum(item.get('rating') == '도움 안 됨' for item in questions)}건")
    metrics[3].metric("도움 됨 👍", f"{sum(item.get('rating') == '도움 됨' for item in questions)}건")
    if not questions:
        st.info("아직 저장된 질문이 없습니다.")
        return
    left, right = st.columns([2, 1])
    keyword = left.text_input("질문에서 찾기")
    status = right.selectbox("보기", ["전체", "도움 됨", "도움 안 됨", "평가 없음"])
    visible = [item for item in questions if (not keyword or keyword.casefold() in item["question"].casefold()) and (status == "전체" or (status == "평가 없음" and item["rating"] is None) or item["rating"] == status)]
    rows = [{"시각": item["created_at"], "질문": item["question"], "평가": item["rating"] or "-", "조건": ", ".join(f"{k}: {'/'.join(v) if isinstance(v, list) else v}" for k, v in item["filters"].items()), "추천한 강의": " / ".join(item["recommended_courses"]), "검색된 강의 수": item["searched_count"], "1위 유사도": item["top_similarity"]} for item in visible]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=520, hide_index=True)


def main() -> None:
    """사이드바 메뉴에 따라 대시보드 페이지를 전환한다."""
    inject_styles()
    st.session_state.courses = load_courses()
    st.session_state.turns = st.session_state.get("turns", [])
    try:
        st.session_state.vectorstore = load_store()
    except Exception as error:
        st.error(f"FAISS 인덱스를 불러오지 못했습니다: {error}")
        st.stop()
    with st.sidebar:
        st.markdown("## 🎓 강의 도우미")
        page = st.radio("메뉴", ["추천", "수집 데이터", "질문 기록"], label_visibility="collapsed")
        st.divider()
    if page == "추천":
        filtered, state = course_filters(st.session_state.courses)
        render_recommendation_page(filtered, state)
    elif page == "수집 데이터":
        render_data_page(st.session_state.courses)
    else:
        render_question_page()


if __name__ == "__main__":
    main()
