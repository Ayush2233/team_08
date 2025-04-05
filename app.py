import json
from agents.eligiblity import EligibilityAgent
from agents.checklist import SubmissionChecklistGenerator
from utils.fileparser import parse_pdf

def main():
    # Path to the attached RFP PDF
    rfp_pdf_path = "./data/IN-ELIGIBLE_RFP.pdf"

    generator = SubmissionChecklistGenerator(rfp_pdf_path)
    final_checklist = generator.execute()
    print("Final Combined Checklist:")
    print(json.dumps(final_checklist, indent=2))
    
    # # Extract full text from the PDF
    # rfp_text = parse_pdf(rfp_pdf_path)
    # print("Extracted RFP Text Preview:")
    # print(rfp_text[:500])  # Debug preview

    # # Initialize the EligibilityAgent and execute the full workflow
    # eligibility_agent = EligibilityAgent()
    # final_report = eligibility_agent.execute(rfp_text)
    
    # print("\nFinal Detailed Eligibility Report with Feedback:")
    # print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    main()
