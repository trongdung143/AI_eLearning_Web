from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là ANALYSIS agent.

            Nhiệm vụ:
            - Phân tích yêu cầu.
            - Trình bày lại những công việc ngắn gọn để những agents sau có thể dựa vào đó mà thực hiện.
            
            Quan trọng:
            - Trình bày các công việc theo đúng thứ tự: công việc nào phải thực hiện trước thì đặt lên trước, công việc nào phụ thuộc thì đặt sau.
            - Chỉ đưa ra những công việc mà agent sau phải thực hiện để giải quyết yêu cầu.
            - Chỉ phân tích, tuyệt đối không tự giải quyết yêu cầu.
            - Không được suy đoán hay gợi ý người dùng cần cung cấp thêm thông tin.
            - Không đưa ra bất kỳ nhận xét hay giải thích gì thêm.
            - Những công việc phải rõ ràng để thực hiện.
            """
        ),
        MessagesPlaceholder("task"),
    ]
)


prompt_supervisor = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            Bạn là SUPERVISOR agent.

            Nhiệm vụ:
            - Đọc yêu cầu và kết quả phân tích analyst vừa thực hiện.
            - Nếu phân tích còn thiếu, mơ hồ, sai sót thì viết feedback và đặt 'next_agent' là 'llm_node'
            - Nếu cần thêm thông tin từ người dùng thì đặt 'next_agent' là 'human_node' 
            - Nếu yêu cầu đã rõ ràng mà không cần phân tích lại hoặc thêm thông tin từ người dùng thì 'next_agent' là '__end__'
            """
        ),
        MessagesPlaceholder("supervision"),
    ]
)
