from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là PLANNER agent.
            Nhiệm vụ:
            - Lập kế hoạch nhiều bước để giải quyết một yêu cầu.
        """
        ),
        MessagesPlaceholder("task"),
    ]
)
