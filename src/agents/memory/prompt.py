from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("task"),
        HumanMessage(
            content="""
            Bạn là MEMORY agent.

            Nhiệm vụ:
            - Tóm tắt toàn bộ hội thoại vừa diễn ra, bao gồm cả nội dung của người dùng (User) và phản hồi của AI.
            - Ghi nhớ tất cả thông tin (bối cảnh, yêu cầu, câu trả lời, quyết định).
            - Giữ tóm tắt ngắn gọn, rõ ràng, nhưng đủ ý để hiểu lại toàn bộ hội thoại trong tương lai.

            Output (chỉ bản tóm tắt, không thêm giải thích khác):
            ### USER
            <tóm tắt các phát ngôn từ user>
            ### AI
            <tóm tắt các phản hồi từ AI>
            """
        ),
    ]
)
