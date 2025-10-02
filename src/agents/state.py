from langgraph.graph import MessagesState
from typing import TypedDict
from pydantic import Field, BaseModel
from langchain_core.messages import BaseMessage
import os


class State(MessagesState):
    thread_id: str
    human: bool | None
    next_agent: str | None
    prev_agent: str | None
    tasks: dict[str, list[str]] | None
    results: dict[str, list[str]] | None
    assigned_agents: dict[str, list[str]] | None
    file_path: str | None
