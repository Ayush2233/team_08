import json
from agents.eligiblity import EligibilityAgent
from utils.fileparser import parse_pdf
from utils.text_utils import chunk_text  # or dynamic_chunk_text if preferred

def main():
    # Path to the attached RFP PDF
    rfp_pdf_path = "./data/IN-ELIGIBLE_RFP.pdf"
    
    # Extract text from the PDF
    rfp_text = parse_pdf(rfp_pdf_path)
    
    # Chunk the text into manageable pieces (using word-count based chunking here)
    sections = chunk_text(rfp_text, chunk_size=500)
    
    eligibility_agent = EligibilityAgent()
    all_mandatory = []
    all_optional = []

    for idx, section in enumerate(sections):
        print(f"\nProcessing Section {idx+1}:")
        result = eligibility_agent.hybrid_eligibility_check(section)
        print("LLM Extracted Response for Section:", result)

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
    
    final_criteria = {
        "mandatory": final_mandatory,
        "optional": final_optional
    }
    
    print("\nFinal Aggregated Eligibility Criteria JSON:")
    print(json.dumps(final_criteria, indent=2))
    
    # Now, generate a detailed eligibility report by comparing the aggregated criteria with the company data.
    report = eligibility_agent.generate_eligibility_report(final_criteria)
    
    print("\nDetailed Eligibility Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
