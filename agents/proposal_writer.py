import google.generativeai as genai
import json
import re
import time
from config import GEMINI_API_KEY, COMPNY_JSON
from utils.fileparser import load_json
import spacy


class ProposalWriterAgent:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")
    
    def call_gemini(self, prompt, retries=5, delay=41, temperature=0.1):
        for attempt in range(retries):
            try:
                response = self.gemini.generate_content(prompt, generation_config={"temperature": temperature})

                # Manually check candidates (if exists)
                if not response.candidates:
                    print("No candidates returned. Skipping chunk.")
                    return ""  # Skip this chunk safely

                # Optional: Check finish reason if you want to be extra strict
                if response.candidates[0].finish_reason == "RECITATION":
                    print("Chunk blocked due to recitation. Skipping.")
                    return ""

                # Safe access
                if hasattr(response, "text"):
                    return response.text.strip()
                elif hasattr(response, "parts"):
                    return "".join([p.text for p in response.parts])
                else:
                    return str(response)

            except Exception as e:
                # Handle blocked chunks gracefully
                if "requires the response to contain a valid `Part`" in str(e) or "finish_reason" in str(e):
                    print(f"Chunk blocked by Gemini (recitation/copyright). Skipping.")
                    return ""
                else:
                    print(f"Gemini API call error: {e}. Retrying in {delay} seconds (Attempt {attempt + 1}/{retries})...")
                    time.sleep(delay)
                    delay *= 2

        print("All retries failed. Skipping this chunk.")
        return ""

    def generate_proposal(self, eligibility_report, checklist, risk_report, detailed_report):
        
        prompt = f"""
        You are a professional proposal writer for government RFP responses. Based on the extracted information below, generate a fully formatted proposal document that is:

        ✅ Legally accurate  
        ✅ Formally structured (following government proposal standards)  
        ✅ Well-organized using Markdown (headings, bullet points, numbered steps)  
        ✅ Suitable for conversion into Word or PDF format  

        ---

        ### 📊 Inputs

        **1. Eligibility Report**:
        {json.dumps(eligibility_report, indent=2)}

        **2. Submission Checklist**:
        {json.dumps(checklist, indent=2)}

        **3. Risk Assessment**:
        {json.dumps(risk_report, indent=2)}

        **4. Recommendations & Notes**:
        {detailed_report}

        ---

        ### 📄 Output Format (Markdown Style)

        Your output should follow this structure:

        ```markdown
        # Proposal for [RFP Title or Opportunity]

        ## 1. Cover Letter
        A short formal letter addressed to the agency.

        ## 2. Executive Summary
        Brief overview of the proposal and the value we offer.

        ## 3. Company Qualifications
        Summarize why the company is eligible, with bullet points.

        ## 4. Scope of Work / Project Approach
        What we propose to do, broken into clear steps or phases.

        ## 5. Compliance & Certifications
        Reference eligibility and legal certifications met (e.g., SAM registration, E-Verify, ISO).

        ## 6. Risk Mitigation
        How we address key risks, as per the RFP’s concerns.

        ## 7. Submission & Timeline
        Refer to the checklist and state readiness to submit with deadlines.

        ## 8. Contact & Signature
        Contact name, title, email, phone."""

        proposal_text = self.call_gemini(prompt, temperature=0.3)
        return proposal_text
        
