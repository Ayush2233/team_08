import re

def chunk_text(text: str, chunk_size: int = 1000) -> list:

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def dynamic_chunk_text(text: str) -> list:
    
    pattern = re.compile(r'(?=\n?\d+\.\s+[A-Z])')
    
    sections = pattern.split(text)
    
    sections = [section.strip() for section in sections if section.strip()]
    return sections