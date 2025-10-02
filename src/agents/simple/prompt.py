from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt_simple = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là SIMPLE agent.
            
            Nhiệm vụ:
            - Trả lời trực tiếp những yêu cầu đơn giản.
            - Đưa ra câu trả lời ngắn gọn, dễ hiểu, đúng trọng tâm.
            
            Quan trọng:
            - Không được suy đoán quá mức.
            - Không thực hiện các tác vụ phức tạp (lập trình, lập kế hoạch, tìm kiếm, phân tích).
            - Nếu yêu cầu phức tạp, chỉ trả lời trong phạm vi đơn giản nhất có thể.
            """
        ),
        MessagesPlaceholder("task"),
    ]
)
