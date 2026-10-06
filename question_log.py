"""질문, 추천 결과, 사용자 평가를 JSONL 파일에 보관한다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "questions.jsonl"


def _read_valid_rows() -> list[dict]:
    """JSONL에서 정상적으로 끝까지 저장된 기록만 읽는다.

    브라우저 새로고침과 파일 저장이 겹치면 마지막 행이 잠시 미완성일 수 있다.
    그 한 행 때문에 질문 기록 화면 전체가 멈추지 않게 한다.
    """
    if not LOG_PATH.exists():
        return []

    rows: list[dict] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # 다음 저장에서 완성될 수 있는 마지막 미완성 행은 건너뛴다.
            continue
    return rows


def load_questions() -> list[dict]:
    """저장된 질문을 최신순으로 불러온다."""
    return list(reversed(_read_valid_rows()))


def save_question(question: str, filters: dict, matches: list[dict]) -> str:
    """새 질문과 검색 결과 요약을 파일에 한 줄로 저장한다."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record_id = str(uuid4())
    record = {
        "id": record_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": question,
        "filters": filters,
        "recommended_courses": [match["metadata"]["title"] for match in matches[:2]],
        "search_results": [
            {"title": match["metadata"]["title"], "similarity": round(match["similarity"], 3)}
            for match in matches
        ],
        "searched_count": len(matches),
        "top_similarity": round(matches[0]["similarity"], 3) if matches else None,
        "rating": None,
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record_id


def update_rating(record_id: str, rating: str) -> None:
    """특정 질문에 대한 도움 됨 또는 도움 안 됨 평가를 갱신한다."""
    if rating not in {"도움 됨", "도움 안 됨"} or not LOG_PATH.exists():
        return
    rows = _read_valid_rows()
    for row in rows:
        if row["id"] == record_id:
            row["rating"] = rating
            break
    LOG_PATH.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
