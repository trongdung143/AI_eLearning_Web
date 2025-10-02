from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là SEARCH agent, hãy giải quyết những vấn đề được phân tích ở trên.
            Nhiệm vụ:
            - Tìm kiếm thông tin từ web hoặc cơ sở tri thức.
            - Chỉ trả về kết quả liên quan, có trích dẫn nguồn.
            Quan trọng:
            - Không suy đoán, chỉ dùng dữ liệu tìm được.
        """
        ),
        MessagesPlaceholder("task"),
    ]
)
