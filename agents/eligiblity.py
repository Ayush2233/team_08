from models.llm import LLMModel
from utils.fileparser import load_json,parse_docx
from config import COMPANY_DATA_PATH
import spacy

from models.llm import LLMModel
from utils.fileparser import load_json, parse_docx
from config import COMPANY_DATA_PATH,COMPNY_JSON
import spacy
import json
import re

class EligibilityAgent:
    def __init__(self):
        self.llm = LLMModel()
        self.company_data = load_json(COMPNY_JSON)


    def hybrid_eligibility_check(self, rfp_text: str) -> dict:
        
        llm_prompt = (
            "You are an expert in government RFP analysis. Your task is to analyze the following RFP document "
            "and extract only the eligibility requirements that an organization must meet to be considered qualified "
            "to submit a proposal. Do not include any general application instructions, guidelines, or points that are "
            "merely suggestions for how to apply. Only extract requirements that are specific, objective criteria (such as "
            "required licenses, certifications, registrations, experience, business structure, or other parameters that "
            "determine eligibility no other details like 'Provide clear and concise responses to all questions' these are not eligiblity parameters these are just guidelines.,).\n\n"
            "example for eligiblity requirements: 'The vendor must be SAM registered and possess an ISO 9001 certification.',' Applicants should have at least 5 years of experience in government contracting' ,'be located in Texas or maintain a branch office in the state.'"
            "Return your result as a valid JSON object with exactly two keys: 'mandatory' and 'optional'. "
            "'mandatory' should list only those requirements that are essential, and 'optional' should list requirements that are desirable but not essential. "
            "Do not include any additional text or commentary. If no requirements are found for a category, return an empty array for that key.\n\n"
            "Do NOT include any general application instructions, formatting guidelines, or other non-eligibility details.\n\n"
            "Example output format:\n"
            "{\n"
            '  "mandatory": ["Requirement 1", "Requirement 2"],\n'
            '  "optional": ["Requirement A", "Requirement B"]\n'
            "}\n\n"
            "Now, analyze the following RFP document and output ONLY the JSON object strictly:\n\n"
            f"{rfp_text}"
        )
        
        llm_response = self.llm.generate(llm_prompt)

        # Handle streaming or non-string responses
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
            import re
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

    def generate_eligibility_report(self, aggregated_json: dict) -> dict:

        # Try to parse company_data as JSON; if not, use it as text.
        try:
            company_json = json.loads(self.company_data)
        except Exception:
            company_json = self.company_data  # Use as-is if not JSON

        report_prompt = (
            "You are an expert in government RFP eligibility evaluation. Below are two JSON objects.\n\n"
            "The first JSON object represents the aggregated eligibility requirements extracted from an RFP, with two keys: "
            "'mandatory' and 'optional'.\n\n"
            "The second JSON object represents the company's data (provided by the company).\n\n"
            "Compare the company's data against the eligibility requirements. Determine if the company meets all mandatory requirements, "
            "and identify which optional requirements are met. Provide a detailed report explaining your conclusions.\n\n"
            "Return ONLY a valid JSON object with the following keys:\n"
            '  "eligible": true or false,\n'
            '  "report": "Detailed explanation",\n'
            '  "missing_mandatory": [list of missing mandatory requirements],\n'
            '  "met_optional": [list of optional requirements that are met]\n\n'
            "Do NOT include any extra text. If a requirement is not met, list it under missing_mandatory. "
            "If the company meets an optional requirement, list it under met_optional.\n\n"
            "Aggregated Eligibility Criteria (from RFP):\n"
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