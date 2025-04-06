import google.generativeai as genai
import json
import re
import time
from config import GEMINI_API_KEY, COMPNY_JSON
from utils.fileparser import load_json
import spacy

class EligibilityAgent:
    def __init__(self):
        # Configure Gemini API with your API key and initialize Gemini model
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")
        self.company_data = load_json(COMPNY_JSON)

    def call_gemini(self, prompt, retries=5, delay=41, temperature=0):
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


    
    def hybrid_eligibility_check(self, rfp_text: str) -> dict:
        llm_prompt = (
            "You are an expert in government RFP analysis. Your task is to analyze the following RFP document and extract ONLY the "
            "objective eligibility requirements that an organization must meet to be considered qualified to submit a proposal. "
            "Ignore any general application instructions, references to attachments, forms, or guidelines (e.g., 'Complete Attachment 1, "
            "'Company Identification Form', 'References and Experience', 'Meet the definition of a business entity as defined in ...'). "
            "Focus solely on requirements that directly indicate measurable or verifiable criteria, such as required licenses, "
            "certifications, registrations, specific experience (e.g., number of years in a relevant field), business structure, "
            "and other parameters that directly impact a company's eligibility.\n\n"
            "Return your result as a valid JSON object with exactly two keys: 'mandatory' and 'optional'. "
            "'mandatory' should list only those requirements that are essential for eligibility, and 'optional' should list requirements "
            "that are desirable but not essential. Do NOT include any extra commentary, instructions, or general guidelines.\n\n"
            "Example output format:\n"
            "{\n"
            '  "mandatory": ["Requirement 1", "Requirement 2"],\n'
            '  "optional": ["Requirement A", "Requirement B"]\n'
            "}\n\n"
            "Now, analyze the following RFP document and output ONLY the JSON object strictly:\n\n"
            f"{rfp_text}"
        )
        
        raw_response = self.call_gemini(llm_prompt)
        print("Gemini Response for Eligibility Extraction:")
        print(raw_response)
        
        try:
            structured_response = json.loads(raw_response)
        except Exception as e:
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    structured_response = json.loads(json_str)
                except Exception as e2:
                    structured_response = {
                        "error": f"Failed to parse JSON from extracted substring: {e2}",
                        "raw_response": raw_response
                    }
            else:
                structured_response = {
                    "error": f"Failed to parse JSON from Gemini response: {e}",
                    "raw_response": raw_response
                }
        return structured_response

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 250) -> list:
        """
        Splits the text into overlapping chunks.
        Each chunk contains `chunk_size` words, with `overlap` words shared with the next chunk.
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move start forward but leave some overlap
            start += chunk_size - overlap

        return chunks


    def aggregate_eligibility_criteria(self, full_rfp_text: str) -> dict:
        """
        Splits the full RFP text into chunks, processes each chunk to extract eligibility criteria,
        and aggregates the results into a single JSON object.
        """
        chunks = self.chunk_text(full_rfp_text, chunk_size=500)
        all_mandatory = []
        all_optional = []
        
        for idx, chunk in enumerate(chunks):
            print(f"\nProcessing Chunk {idx+1}:")
            result = self.hybrid_eligibility_check(chunk)
            print("Extracted Response:", result)
            
            if "mandatory" in result and "optional" in result:
                if result["mandatory"]:
                    all_mandatory.extend(result["mandatory"])
                if result["optional"]:
                    all_optional.extend(result["optional"])
            else:
                fallback = result.get("llm_criteria", [])
                if fallback:
                    all_mandatory.extend(fallback)
        
        final_mandatory = list(set(all_mandatory))
        final_optional = list(set(all_optional))
        
        aggregated = {
            "mandatory": final_mandatory,
            "optional": final_optional
        }
        return aggregated

    def generate_eligibility_report(self, aggregated_json: dict) -> dict:
        """
        Compare the aggregated eligibility criteria with the company's data and generate a detailed report.
        Return a valid JSON object with the following keys:
          - "eligible": true or false,
          - "report": Detailed explanation,
          - "missing_mandatory": [list of missing mandatory requirements],
          - "met_optional": [list of optional requirements that are met]
        """
        try:
            company_json = self.company_data
        except Exception:
            company_json = self.company_data

        report_prompt = (
            "You are an expert in government RFP eligibility evaluation. Below are two JSON objects.\n\n"
            "The first JSON object represents the aggregated eligibility requirements extracted from an RFP. It contains two keys:\n"
            "'mandatory' - essential requirements that must be met for eligibility,\n"
            "'optional' - desirable but non-essential requirements.\n\n"
            "The second JSON object represents the company's eligibility data, describing qualifications, certifications, registrations, "
            "and other information the company has provided.\n\n"
            "Your task is to compare these two objects and determine whether the company meets ALL mandatory requirements, and identify which optional "
            "requirements (if any) are satisfied.\n\n"
            "Important: Some eligibility requirements may be conditional or situational. For example, a requirement may only apply if the company is "
            "claiming a specific status (e.g., MBE/WBE certification). If the company data clearly states that a condition does not apply to it, "
            "treat the related requirement as **not applicable** and do not count it as missing.\n\n"
            "Apply logical reasoning and ignore requirements that are clearly waived or irrelevant based on company context.\n\n"
            "Return ONLY a valid JSON object with the following keys:\n"
            '  "eligible": true or false,\n'
            '  "report": "Detailed explanation",\n'
            '  "missing_mandatory": [list of missing mandatory requirements],\n'
            '  "met_optional": [list of optional requirements that are met]\n\n'
            "DO NOT include any commentary, formatting, or extra text outside of the JSON.\n\n"
            "Aggregated Eligibility Criteria (from RFP):\n"
            f"{json.dumps(aggregated_json)}\n\n"
            "Company Data:\n"
            f"{json.dumps(company_json)}\n\n"
            "Output ONLY a valid JSON object strictly."
        )

        raw_response = self.call_gemini(report_prompt)
        print("Gemini Raw Response for Report:")
        print(raw_response)
        
        try:
            report = json.loads(raw_response)
        except Exception as e:
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    report = json.loads(json_str)
                except Exception as e2:
                    report = {"error": f"Failed to parse report JSON: {e2}", "raw_response": raw_response}
            else:
                report = {"error": f"Failed to parse report JSON: {e}", "raw_response": raw_response}
        return report

    def execute(self, full_rfp_text: str) -> dict:
        """
        Executes the full eligibility evaluation workflow:
         1. Aggregates eligibility criteria from the RFP.
         2. Generates a detailed eligibility report comparing the aggregated criteria with company data.
         3. Returns the final report.
        """
        aggregated_criteria = self.aggregate_eligibility_criteria(full_rfp_text)
        print("\nFinal Aggregated Eligibility Criteria JSON:")
        print(json.dumps(aggregated_criteria, indent=2))
        
        report = self.generate_eligibility_report(aggregated_criteria)
        return report

# # Example main flow to test the agent
# if __name__ == "__main__":
#     # For testing, you can use sample text or load from a file.
#     test_text = (
#         "The vendor must be SAM registered and possess an ISO 9001 certification. "
#         "Applicants should have at least 5 years of experience in government contracting and "
#         "be located in Texas or maintain a branch office in the state."
#     )
#     agent = EligibilityAgent()
#     results = agent.execute(test_text)
#     print(json.dumps(results, indent=2))
