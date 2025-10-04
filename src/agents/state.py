from langgraph.graph import MessagesState


class State(MessagesState):
    user_id: str
    lesson_id: str
    type_request: str
    task: str
    result: str