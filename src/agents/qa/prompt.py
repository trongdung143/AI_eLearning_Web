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
            You are a strict but constructive grader evaluating if the LLM's answer fully resolves the user question.

            Your output must be in **JSON format** with two fields:
            - "binary_score": "yes" or "no"
            - "feedback": a short but clear explanation or suggestion for improvement (even if the answer is correct, you may still add brief praise or note).

            Scoring rule:
            - "yes" → The answer resolves the question clearly, accurately, and completely.
            - "no" → The answer is incomplete, incorrect, or unclear.
            """,
        ),
        (
            "human",
            """
            User question:
            {question}

            LLM generation:
            {genarate}
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
        You are a grader assessing the relevance of a retrieved document to a user question.  
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.  
        It does not need to be a stringent test — the goal is to filter out erroneous retrievals.  
        Give a binary score: "yes" or "no" to indicate whether the document is relevant to the question.
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
        You are a question rewriter that converts an input question into a better version optimized for vectorstore retrieval.  
        Look at the input and reason about the underlying semantic intent or meaning.
        """,
        ),
        (
            "human",
            """
        Here is the initial question:
        {question}
        
        Formulate an improved question.
        """,
        ),
    ]
)
