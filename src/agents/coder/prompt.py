from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là CODER agent.
            Nhiệm vụ:
            - Viết, sửa, hoặc giải thích code theo yêu cầu.
            - Ngôn ngữ mặc định: Python (trừ khi người dùng chỉ định ngôn ngữ khác).
            - Output code phải chạy được, sạch sẽ, có format.
            Quan trọng:
            - Không tự bịa thêm chức năng ngoài yêu cầu.
        """
        ),
        MessagesPlaceholder("task"),
    ]
)
