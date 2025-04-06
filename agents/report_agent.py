import json
import re
import time
import google.generativeai as genai
import google.api_core.exceptions

class DetailedReportAgent:

    def call_gemini(self, prompt, retries=5, delay=41):
        """Calls the Gemini API with retries for ResourceExhausted errors."""
        model = genai.GenerativeModel("gemini-2.0-flash")
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"Quota exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"Error: {e}")
                break
        print("All retries failed. Please check your API quota.")
        return ""

    def generate_formatted_report(self, eligibility_report: dict, checklist: dict,risk_assessment) -> str:
        
        prompt = (
            "You are an expert in government RFP analysis and proposal writing. "
            "Below are three JSON objects:\n\n"
            "1. Aggregated Eligibility Requirements extracted from an RFP (with keys 'mandatory' and 'optional').\n\n"
            "2. Submission Checklist extracted from the RFP (detailing document formatting, attachments, deadlines, etc.).\n\n"
            "3. Risk assessment about risky clauses and suggestions to balance terms for organizations"
            "Your task is to produce a very detailed report in Markdown format that includes the following sections:\n\n"
            "## 1. Eligibility Analysis\n"
            "- Identify any deal-breakers early by highlighting any missing mandatory requirements.\n"
            "- Summarize the must-have qualifications, certifications, and experience needed to bid.\n\n"
            "## 2. Submission Checklist\n"
            "- List and explain all submission requirements such as document format (page limits, font type/size, line spacing, TOC requirements), "
            "required attachments, and deadlines.\n\n"
            "## 3. Recommendations\n"
            "- Provide actionable suggestions (todos) on how to address any issues or gaps.\n"
            "- List pitfalls or things to avoid (don'ts) when preparing the proposal.\n\n"
            "4. Risk Assessment "
            "-Risky clauses and red flags where the org needs to be aware and have to balance terms (explain each in bit detail)"
            "Ensure that the report is comprehensive, well-structured with clear headings, bullet points, and paragraphs. "
            "If essential information is missing, include a note stating that further verification is required.\n\n"
            "Do NOT include any extra text or commentary beyond the formatted beautiful report with icons etc.\n\n"
            "Make sure to make the report readble in sections starting with whether company is eligible or not then other details in points with formatting."
            "Aggregated Eligibility Requirements:\n"
            f"{json.dumps(eligibility_report, indent=2)}\n\n"
            "Submission Checklist:\n"
            f"{json.dumps(checklist, indent=2)}\n\n"
             "Risk Assessment:\n"
            f"{json.dumps(risk_assessment, indent=2)}\n\n"
        )
        llm_response = self.call_gemini(prompt=prompt,retries=5,delay=41)
        # Assuming the response is plain text Markdown.
        return llm_response
