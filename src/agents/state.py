from langgraph.graph import MessagesState


class State(MessagesState):
    thread_id: str
    lesson_id: str
    type_request: str
    task: str
    result: str
    file_path: str | None
