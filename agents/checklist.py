import json
import re
import time
import google.generativeai as genai
import google.api_core.exceptions
from config import GEMINI_API_KEY, COMPNY_JSON, OUTPUT_FILE
from utils.fileparser import load_rfp_text


class SubmissionChecklistGenerator:
    def __init__(self, rfp_file):
        self.rfp_file = rfp_file
        genai.configure(api_key=GEMINI_API_KEY)

    def load_rfp(self):
        """Loads the full RFP text from the specified file."""
        return load_rfp_text(self.rfp_file)

    def chunk_rfp_text(self, rfp_text, max_chunk_size=2000):
        """Splits the RFP text into chunks of max_chunk_size characters."""
        chunks = []
        start = 0
        while start < len(rfp_text):
            end = start + max_chunk_size
            chunks.append(rfp_text[start:end])
            start = end
        return chunks

    def build_prompt(self, rfp_text):
        """Builds a Gemini prompt using the provided RFP text."""
        prompt = f"""
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
        \"\"\"{rfp_text}\"\"\"
        """
        return prompt

    def call_gemini(self, prompt, retries=5, delay=41):
        """Calls the Gemini API with retries for ResourceExhausted errors."""
        model = genai.GenerativeModel("gemini-2.0-flash")
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"❌ Quota exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        print("❌ All retries failed. Please check your API quota.")
        return ""

    def clean_response(self, text):
        """Cleans and extracts the JSON array from the Gemini response."""
        text = text.strip()
        # Remove Markdown formatting if present.
        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.strip("`").strip()

        # Extract JSON array using regex.
        match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def parse_response(self, text):
        """Parses the cleaned response text as JSON."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return None

    def save_checklist(self, checklist):
        """Saves the final checklist JSON to the OUTPUT_FILE."""
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(checklist, f, indent=2)
            print(f"✅ Checklist saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Error saving checklist: {e}")

    def generate_checklist(self):
        """Executes the workflow for generating the submission checklist."""
        rfp_text = self.load_rfp()
        print("📄 RFP Loaded. Length:", len(rfp_text))
        chunks = self.chunk_rfp_text(rfp_text)
        print(f"🧠 Split RFP into {len(chunks)} chunks.")

        full_checklist = []
        for i, chunk in enumerate(chunks):
            print(f"\n🔍 Processing chunk {i+1} of {len(chunks)}...")
            prompt = self.build_prompt(chunk)
            print("🚀 Calling Gemini API...")
            response_text = self.call_gemini(prompt)
            if not response_text.strip():
                print(f"❌ Empty response from Gemini for chunk {i+1}. Skipping...")
                continue
            print("🧹 Cleaning response...")
            cleaned = self.clean_response(response_text)
            checklist = self.parse_response(cleaned)
            if checklist is None:
                print(f"❌ Failed to parse JSON for chunk {i+1}.")
                continue
            full_checklist.extend(checklist)
            print("✅ Chunk processed successfully.")

        return full_checklist

    def execute(self):
        """
        Executes the full workflow:
          1. Generate the submission checklist from the RFP.
          2. Save the combined checklist.
        Returns the final checklist.
        """
        checklist = self.generate_checklist()
        self.save_checklist(checklist)
        return checklist

# def main():
#     generator = SubmissionChecklistGenerator(RFP_FILE)
#     final_checklist = generator.execute()
#     print("\nFinal Combined Checklist:")
#     print(json.dumps(final_checklist, indent=2))

# if __name__ == "__main__":
#     main()
