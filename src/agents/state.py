from langgraph.graph import MessagesState


class State(MessagesState):
    type_request: str
    task: str | None
    result: str | None
    lecture: dict[str, tuple[str, list[str]]] | None
    quiz: dict[str, list[str]] | None
    document_path: str | None
    lesson_id: str | None
