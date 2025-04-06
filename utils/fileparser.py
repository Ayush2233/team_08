import json
import docx
import fitz
import pdfplumber

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


def load_rfp_text(file_path):
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text


def parse_pdf_streamlit(file_obj):
    """
    Parse PDF text from a file-like object (e.g., uploaded file from Streamlit).
    """
    try:
        # Read the file bytes from the file-like object
        file_bytes = file_obj.read()
        # Open the document from bytes. The "pdf" argument specifies the file type.
        doc = fitz.open("pdf", file_bytes)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return full_text
    except Exception as e:
        raise Exception(f"Error parsing PDF: {e}")