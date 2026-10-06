"""courses.json 변경 사항을 FAISS 인덱스에 반영하는 실행 프로그램."""

from rag import sync_vectorstore


def main() -> None:
    """증분 인덱싱 결과를 실행 화면에 출력한다."""
    result = sync_vectorstore()
    print("FAISS 인덱스 갱신 완료")
    print(f"추가: {result['added']}건")
    print(f"수정: {result['updated']}건")
    print(f"삭제: {result['deleted']}건")
    print(f"변경 없음: {result['unchanged']}건")


if __name__ == "__main__":
    main()
