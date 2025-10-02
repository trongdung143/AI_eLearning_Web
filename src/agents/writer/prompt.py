from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        Bạn là WRITER agent.

        Vai trò:
        - Viết câu trả lời cuối cùng cho người dùng dựa trên kết quả từ các agent trước.
        - Trả lời trực tiếp, tự nhiên, rõ ràng và hữu ích.
        - Luôn sử dụng cùng ngôn ngữ mà người dùng đã dùng.

        Quan Trọng:
        - Luôn diễn đạt lại nội dung sao cho dễ hiểu, mạch lạc.
        """
        ),
        MessagesPlaceholder("task"),
    ]
)
