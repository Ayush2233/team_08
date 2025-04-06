import google.generativeai as genai
import json
import time
from config import GEMINI_API_KEY

class RiskAssessmentAgent:
    def __init__(self):
        # Initialize Gemini API with your API key
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
    
    def chunk_text(self, text, chunk_size=2000):
        """
        Splits the input text into chunks of approximately 'chunk_size' characters.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end
        return chunks

    def call_gemini(self, prompt, retries=5, delay=41):
        """
        Calls the Gemini API with retries for ResourceExhausted errors.
        """
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                # Optionally, check for ResourceExhausted error specifically
                print(f"Quota exceeded or error encountered. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
        print("All retries failed. Please check your API quota.")
        return ""

    def make_gemini_prompt(self, clause):
        """
        Builds the Gemini prompt using your specified prompt format.
        """
        prompt = f"""
You are a legal analyst AI reviewing contract clauses and performing risk assessment. Review the below contract 
in terms of contract risks and identify biased clauses that could put an organization at a disadvantage (e.g., unilateral termination rights).
Suggest modifications to balance contract terms (e.g., adding a notice period for termination).

Contract: "{clause}"

Return ONLY a JSON in this format:

{{
  "final_risk": "<Best fitting risk category>",
  "justification": "<Short explanation>",
  "suggestions": "<Suggestions to balance contract terms>"
}}
"""
        return prompt

    def analyze_clause(self, clause):
        """
        Analyzes a single clause by sending it to the Gemini API (with retry logic) and returns the parsed JSON result.
        """
        prompt = self.make_gemini_prompt(clause)
        response_text = self.call_gemini(prompt)
        try:
            result = json.loads(response_text.strip())
        except Exception as e:
            result = {
                "error": f"Failed to parse JSON: {e}",
                "raw_response": response_text.strip()
            }
        return result

    def execute(self, pdf_text):
        """
        Executes the full risk assessment workflow:
         1. Chunks the PDF text into manageable pieces.
         2. Analyzes each chunk using the Gemini API.
         3. Aggregates all analysis results into a final JSON object.
        """
        chunks = self.chunk_text(pdf_text, chunk_size=2000)
        aggregated_results = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            if len(chunk.strip()) > 20:  # Ensure the chunk is non-trivial
                result = self.analyze_clause(chunk)
                aggregated_results.append({
                    "clause": chunk,
                    "analysis": result
                })
                time.sleep(1)  # Delay to avoid rate limiting
        
        final_json = {"risk_assessments": aggregated_results}
        return final_json

# # Example main flow to test the agent
# if __name__ == "__main__":
#     # For testing, you can use sample text or load from a file.
#     test_text = (
#         "The vendor must be SAM registered and possess an ISO 9001 certification. "
#         "Applicants should have at least 5 years of experience in government contracting and "
#         "be located in Texas or maintain a branch office in the state."
#     )
#     agent = RiskAssessmentAgent()
#     results = agent.execute(test_text)
#     print(json.dumps(results, indent=2))
