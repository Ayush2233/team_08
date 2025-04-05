import json
import docx
import fitz

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_docx(file_path: str) -> str:
    """
    Parse and return text content from a DOCX file.
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

def parse_rfp_document(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

def parse_pdf(file_path: str) -> str:
    """
    Parse and return text content from a PDF file using PyMuPDF.
    """
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text