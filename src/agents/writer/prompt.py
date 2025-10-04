from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            Viết lại câu trả lời theo phong cách giáo viên đang trả lời cho học sinh.
            """
        ),
        ("human",
         """
            Câu hỏi của học sinh:
            {question}
            
            Câu trả lời:
            {content}
            
            Hãy viết lại câu trả lời trên theo phong cách của giáo viên đang trả lời cho học sinh.
        """
        ),
    ]
)
