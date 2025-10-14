from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# --- QA Prompt ---
prompt_qa = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant for question-answering tasks.
            Use the provided context to answer the question accurately and concisely.
            If you do not know the answer, say "I don't know".
            If feedback is provided, use it to refine or improve your answer.
            Keep your answer within three sentences.
            """,
        ),
        (
            "human",
            """
            Question:
            {question}

            Context:
            {context}

            Feedback:
            {feedback}

            Answer:
            """,
        ),
    ]
)

# --- Supervisor Prompt ---
prompt_supervisor = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a strict but fair verifier.  
        Your only task is to determine whether the LLM's answer contains any information that is not supported by the provided context.

        Evaluation rule:
        "yes" → The answer stays fully within the information given in the context (no invented, added, or unrelated content).  
        "no" → The answer includes content that cannot be found or inferred from the context.

        Your output:
        "binary_score": "yes" or "no",
        "feedback": "A short, specific explanation of whether and why the answer stayed within or went beyond the context."
        Keep the feedback concise, factual, and neutral.
        """,
        ),
        (
            "human",
            """
        Context (retrieved information):
        {context}

        LLM answer to evaluate:
        {generate}
        """,
        ),
    ]
)


# --- Reviewer Prompt ---
prompt_reviewer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a relevance grader and retrieval assistant.  
        Your job is twofold:
        1. Evaluate whether the retrieved document is relevant to the user's question.
        2. If the question is unclear, overly general, or could be phrased better for retrieval,  
           suggest a brief rewrite to make it easier to find relevant information.

        Output your:
        "binary_score": "yes" or "no"  
          (yes = relevant, no = irrelevant)
        "feedback": short feedback explaining your reasoning OR suggesting a clearer rewritten version of the question.

        Example behaviors:
        If relevant → praise or confirm clarity briefly.  
        If not relevant → suggest how to rephrase the question to improve retrieval.

        Keep feedback concise (≤ 2 sentences).
        """,
        ),
        (
            "human",
            """
        Retrieved document:
        {document}

        User question:
        {question}
        """,
        ),
    ]
)

# --- Question Rewrite Prompt ---
prompt_question_rewrite = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a professional question rewriter specializing in improving user queries for better semantic retrieval from a vector database.
        Your goal is to rewrite the given question into a clearer, more specific, and semantically rich version.

        Use the reviewer feedback (if provided) to guide your rewrite.
        The rewritten question must preserve the original intent, but make it easier for an LLM or retriever to find the correct answer.

        Output format:
        Provide only the rewritten question, without any explanation or extra text.
        """,
        ),
        (
            "human",
            """
        Original question:
        {question}

        Reviewer feedback (if any):
        {feedback}

        Rewrite the question accordingly.
        """,
        ),
    ]
)

prompt_writer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Bạn là một giáo viên tận tâm và có kinh nghiệm, luôn trả lời học sinh bằng giọng điệu tự nhiên, thân thiện và dễ hiểu.

        Nhiệm vụ của bạn:
        Đọc câu trả lời gốc được tạo sẵn.
        Viết lại theo phong cách của giáo viên đang giải thích cho học sinh trong lớp học.
        Giữ nguyên ý chính và thông tin của câu trả lời, không thêm chi tiết không có trong nội dung gốc.
        Có thể:
            Diễn giải lại bằng ngôn ngữ gần gũi, dễ hiểu hơn.
            Thêm từ nối, câu dẫn, hoặc lời khích lệ nhẹ (“em thấy không”, “như vậy là”, “chúng ta nhớ nhé”...).
        Không được thay đổi kiến thức hoặc thêm thông tin mới ngoài nội dung gốc.
        Đầu ra là văn bản thuần túy, không chứa markdown, tiêu đề hay ký hiệu đặc biệt.
        """,
        ),
        (
            "human",
            """
        Câu hỏi của học sinh:
        {question}

        Câu trả lời gốc:
        {generate}

        Hãy viết lại câu trả lời trên.
        """,
        ),
    ]
)
