import json
import re
from models.llm import LLMModel
from utils.fileparser import load_json, parse_docx
from config import COMPANY_DATA_PATH, COMPNY_JSON
import spacy

class EligibilityAgent:
    def __init__(self):
        self.llm = LLMModel()
        self.company_data = load_json(COMPNY_JSON)

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
        
        llm_response = self.llm.generate(llm_prompt)
        if hasattr(llm_response, "read"):
            llm_response = llm_response.read()
        elif not isinstance(llm_response, str):
            try:
                llm_response = "".join(list(llm_response))
            except Exception as e:
                llm_response = str(llm_response)
        
        print("LLM Response:")
        print(llm_response)
        
        try:
            structured_response = json.loads(llm_response)
        except Exception as e:
            json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    structured_response = json.loads(json_str)
                except Exception as e2:
                    structured_response = {
                        "error": f"Failed to parse JSON from extracted substring: {e2}",
                        "raw_response": llm_response
                    }
            else:
                structured_response = {
                    "error": f"Failed to parse JSON from LLM response: {e}",
                    "raw_response": llm_response
                }
        return structured_response

    def chunk_text(self, text: str, chunk_size: int = 500) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
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
        Compare the aggregated eligibility criteria with the company's data and generate a detailed report indicating
        whether the company meets all mandatory requirements and which optional requirements are met.
        Return a valid JSON object with the following keys:
          - "eligible": true or false,
          - "report": Detailed explanation,
          - "missing_mandatory": [list of missing mandatory requirements],
          - "met_optional": [list of optional requirements that are met]
        """
        try:
            company_json = self.company_data
        except Exception:
            company_json = self.company_data  # Use as-is if not valid JSON

        report_prompt = (
            "You are an expert in government RFP eligibility evaluation. Below are two JSON objects.\n\n"
            "The first JSON object represents the aggregated eligibility requirements extracted from an RFP, with two keys: "
            "'mandatory' and 'optional'.\n\n"
            "The second JSON object represents the company's eligibility data, which contains the essential parameters the company possesses.\n\n"
            "Compare the company's data against the eligibility requirements. Determine if the company meets ALL the mandatory "
            "requirements, and identify which optional requirements are met. Provide a detailed report explaining your conclusions.\n\n"
            "Return ONLY a valid JSON object with exactly the following keys:\n"
            '  "eligible": true or false,\n'
            '  "report": "Detailed explanation",\n'
            '  "missing_mandatory": [list of missing mandatory requirements],\n'
            '  "met_optional": [list of optional requirements that are met]\n\n'
            "Do NOT include any extra text.\n\n"
            "Aggregated Eligibility Criteria (from RFP):\n"
            "If the company misses most of the information required to validate then add additonal comment in the 'report' key as insufficient data to validate "
            f"{json.dumps(aggregated_json)}\n\n"
            "Company Data:\n"
            f"{json.dumps(company_json)}\n\n"
            "Output ONLY a valid JSON object strictly."
        )
        
        llm_response = self.llm.generate(report_prompt)
        if hasattr(llm_response, "read"):
            llm_response = llm_response.read()
        elif not isinstance(llm_response, str):
            try:
                llm_response = "".join(list(llm_response))
            except Exception as e:
                llm_response = str(llm_response)
        
        print("LLM Raw Response for Report:")
        print(llm_response)
        
        try:
            report = json.loads(llm_response)
        except Exception as e:
            json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    report = json.loads(json_str)
                except Exception as e2:
                    report = {"error": f"Failed to parse report JSON: {e2}", "raw_response": llm_response}
            else:
                report = {"error": f"Failed to parse report JSON: {e}", "raw_response": llm_response}
        return report

    def execute(self, full_rfp_text: str) -> dict:
    
        aggregated_criteria = self.aggregate_eligibility_criteria(full_rfp_text)
        print("\nFinal Aggregated Eligibility Criteria JSON:")
        print(json.dumps(aggregated_criteria, indent=2))
        
        report = self.generate_eligibility_report(aggregated_criteria)
        return report