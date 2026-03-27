import re
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

import re


def clean_txt(txt: str) -> str:
    txt = (
        txt.replace("\\n", " ").replace("\n", " ").replace("\t", " ").replace("\r", " ")
    )

    txt = re.sub(r"[^A-Za-zÀ-ỹ0-9\s]", " ", txt)

    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def format_document(current_page: Document) -> str:
    return current_page.page_content


def extract_content(response) -> str:
    return response.content  # [-1]["text"]
