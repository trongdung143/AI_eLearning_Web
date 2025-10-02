from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are LOGIC/MATH agent.
            Your mision:
            - Giải các bài toán toán học/logic (số học, đại số, giải tích, xác suất, thống kê, đổi đơn vị, phân tích thứ nguyên...).
            - Phân tích từng bước giải bài.
            """
        ),
        MessagesPlaceholder("task"),
    ]
)
