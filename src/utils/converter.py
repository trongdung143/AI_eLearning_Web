from docx2pdf import convert


def word_to_pdf(docx_path: str) -> str:
    pdf_path = docx_path.replace(".docx", ".pdf")
    convert(docx_path, pdf_path)
    return pdf_path
