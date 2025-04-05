import json
import re
from pathlib import Path
import pdfplumber
import google.generativeai as genai
import time

# Constants
RFP_FILE = "RFP1.pdf"
OUTPUT_FILE = "submission_checklist.json"
GEMINI_API_KEY = "AIzaSyBuuJvcXGFPltCUQqwNW6yO-lr4PtHGWQ0"
MAX_CHUNK_SIZE = 1000  # Adjust based on Gemini model limits

# Load RFP text from PDF
def load_rfp_text(file_path):
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

# Split RFP text into manageable chunks
def chunk_rfp_text(rfp_text, max_chunk_size=MAX_CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(rfp_text):
        end = start + max_chunk_size
        chunks.append(rfp_text[start:end])
        start = end
    return chunks

# Build Gemini prompt
def build_prompt(rfp_text):
    return f"""
You are a government contract analyst tasked with extracting a comprehensive and structured checklist of all submission requirements from the following RFP text. Given that this document is large and may contain multiple sections, please ensure that no relevant submission details are missed.

The checklist should include, but is not limited to, the following categories:
- Page limits (for entire proposal, sections, or individual documents)
- Font and formatting rules (font type, size, line spacing, margins, etc.)
- Required attachments (forms, certifications, company data, etc.)
- Deadlines and submission timelines (specific dates or timeframes for submission)
- Required headers, section titles, or naming conventions
- Legal and compliance requirements (e.g., specific certifications, eligibility, or qualifications)
- Special document handling instructions (e.g., multiple copies, electronic submission, etc.)
- Any other specific submission guidelines mentioned throughout the RFP

The extracted checklist should be formatted as a valid JSON array, with each checklist item having:
- "item": A brief title or description of the requirement
- "details": A brief summary of what is required for that item

Return ONLY a valid JSON array with no extra text, explanations, or comments. 

RFP Text:
\"\"\"
{rfp_text}
\"\"\"
"""

# Call Gemini API
# def call_gemini(prompt):
#     genai.configure(api_key=GEMINI_API_KEY)
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     return response.text

import time
import google.api_core.exceptions

def call_gemini(prompt, retries=5, delay=41):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"❌ Quota exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2  # Exponential backoff: increase the delay for each retry
        except Exception as e:
            print(f"❌ Error: {e}")
            break  # Exit the loop on other types of errors

    print("❌ All retries failed. Please check your API quota.")
    return ""


# Clean up Gemini response to extract valid JSON
def clean_response(text):
    text = text.strip()

    # Remove Markdown wrapping if any
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.strip("`").strip()

    # Extract JSON array using regex
    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if match:
        return match.group(0)

    return text

# Save JSON checklist
def save_checklist(json_text, output_file):
    try:
        if isinstance(json_text, str):
            checklist = json.loads(json_text)
        else:
            checklist = json_text
    except json.JSONDecodeError:
        print("❌ Final save: could not decode JSON. Saving raw output.")
        checklist = {"raw_output": json_text}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2)

    print(f"✅ Checklist saved to {output_file}")

# Main execution
def main():
    print("📄 Loading RFP...")
    rfp_text = load_rfp_text(RFP_FILE)

    print("🧠 Splitting RFP into manageable chunks...")
    rfp_chunks = chunk_rfp_text(rfp_text)

    full_checklist = []

    for i, chunk in enumerate(rfp_chunks):
        print(f"\n🔍 Processing chunk {i+1} of {len(rfp_chunks)}...")
        prompt = build_prompt(chunk)

        print("🚀 Calling Gemini API...")
        checklist_text = call_gemini(prompt)

        if not checklist_text.strip():
            print(f"❌ Empty response from Gemini for chunk {i+1}. Skipping...")
            continue

        print("🧹 Cleaning and parsing response...")
        cleaned = clean_response(checklist_text)

        try:
            chunk_checklist = json.loads(cleaned)
            full_checklist.extend(chunk_checklist)
            print("✅ Chunk parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing checklist for chunk {i+1}: {e}")
            debug_path = f"debug_chunk_{i+1}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(checklist_text)
            print(f"📝 Saved raw response to {debug_path} for review.")

    print("\n💾 Saving final combined checklist...")
    save_checklist(full_checklist, OUTPUT_FILE)

if __name__ == "__main__":
    main()
