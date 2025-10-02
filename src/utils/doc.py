import fitz
import os
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import re
import asyncio


def extract_text_from_pdf(data_path: str) -> str:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PDF not found at: {data_path}")

    with fitz.open(data_path) as doc:
        text = "".join(page.get_text() for page in doc)

    return text


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.get_text()
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    return content


async def get_content_web_by_url(url: str):
    loader = RecursiveUrlLoader(
        url,
        max_depth=5,
        extractor=bs4_extractor,
    )
    docs = await asyncio.to_thread(loader.load)
    return docs
