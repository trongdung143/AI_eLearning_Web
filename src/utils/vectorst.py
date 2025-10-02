import asyncio
import os

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.setup import GOOGLE_API_KEY
from src.utils.converter import word_to_pdf
from src.utils.doc import get_content_web_by_url


def remove_duplicate_paragraphs(text: str) -> str:
    paragraphs = text.split("\n\n")
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        para_strip = para.strip()
        if para_strip and para_strip not in seen:
            seen.add(para_strip)
            unique_paragraphs.append(para_strip)
    return "\n\n".join(unique_paragraphs)


async def create_vector_db_from_text(url: str) -> FAISS:
    content = ""
    docs = await get_content_web_by_url(url)
    for doc in docs:
        content += doc.page_content

    lines = content.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip() != ""]
    content = "\n".join(cleaned_lines)
    content = remove_duplicate_paragraphs(content)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(content)

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    db = await FAISS.afrom_texts(texts, embedding)
    return db


async def create_vector_db_from_file(file_path: str, file_base: str):
    vector_db_path = f"src/tools/data/vectorstores/{file_base}"
    if os.path.exists(vector_db_path):
        return None

    extension = os.path.splitext(file_path)[1].lower()
    temp_pdf_created = False

    if extension == ".docx":
        file_path = word_to_pdf(file_path)
        temp_pdf_created = True

    loader = PyPDFLoader(file_path)
    documents = await asyncio.to_thread(loader.load)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    texts = await asyncio.to_thread(text_splitter.split_documents, documents)

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    db = await asyncio.to_thread(FAISS.from_documents, texts, embedding)
    await asyncio.to_thread(db.save_local, vector_db_path)

    if temp_pdf_created:
        os.remove(file_path)

    return db


async def read_vectorstores(file_base: str) -> FAISS:
    vector_db_path = f"src/tools/data/vectorstores/{file_base}"

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    db = await asyncio.to_thread(
        FAISS.load_local,
        vector_db_path,
        embedding,
        allow_dangerous_deserialization=True,
    )
    return db
