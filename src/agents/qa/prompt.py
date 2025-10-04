from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain import hub

# --- QA Prompt ---
prompt_qa = hub.pull("rlm/rag-prompt")

# --- Supervisor Prompt ---
prompt_supervisor = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a grader assessing whether an answer addresses or resolves a question.  
        Give a binary score: "yes" or "no".  
        "Yes" means that the answer resolves the question.
        """
    ),
    (
        "human",
        """
        User question:
        {question}
        
        LLM generation:
        {genarate}
        """
    )
])

# --- Reviewer Prompt ---
prompt_reviewer = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a grader assessing the relevance of a retrieved document to a user question.  
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.  
        It does not need to be a stringent test — the goal is to filter out erroneous retrievals.  
        Give a binary score: "yes" or "no" to indicate whether the document is relevant to the question.
        """
    ),
    (
        "human",
        """
        Retrieved document:
        {document}
        
        User question:
        {question}
        """
    )
])

# --- Question Rewrite Prompt ---
prompt_question_rewrite = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a question rewriter that converts an input question into a better version optimized for vectorstore retrieval.  
        Look at the input and reason about the underlying semantic intent or meaning.
        """
    ),
    (
        "human",
        """
        Here is the initial question:
        {question}
        
        Formulate an improved question.
        """
    )
])
