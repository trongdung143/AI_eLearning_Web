import re
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


def clean_txt(txt: str) -> str:

    if isinstance(txt, list):
        txt = " ".join(str(x) for x in txt)

    if "```json" in txt or "```" in txt:
        txt = txt.replace("```json", "").replace("```", "").strip()
        txt = (
            txt.replace("\\n", " ")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("\r", " ")
        )
    txt = re.sub(r"[^A-Za-zÀ-ỹ0-9\s]", " ", txt)

    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def format_document(current_page: Document) -> str:
    return current_page.page_content
