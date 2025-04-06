import streamlit as st
import os
import json
import pymongo
from utils.fileparser import parse_pdf_streamlit  # For uploaded PDF files
from agents.eligiblity import EligibilityAgent
from agents.checklist import SubmissionChecklistGenerator  # Ensure implemented
from agents.report_agent import DetailedReportAgent
from agents.risk_assessment import RiskAssessmentAgent
from utils.RAGretriver import RAGretriver  # For chatbot functionality
import torch
torch.classes.__path__ = []

def upload_to_mongodb(data):
    try:
        # Replace <db_password> with your actual password or fetch from environment variables
        connection_string = "mongodb+srv://apurva:apurva123@cluster0.0nmuvte.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = pymongo.MongoClient(connection_string)
        db = client["consultadd"]
        collection = db["hackathon"]
        result = collection.insert_one(data)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error uploading to MongoDB: {e}")
        return None



st.set_page_config(page_title="RFP Analysis, Detailed Report & Chatbot", layout="wide")
st.title("RFP Analysis, Detailed Report & Chatbot")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.write("""
**Upload an RFP PDF** and let our system process it.  
In the **Detailed Report** tab, the document is analyzed to produce:  
- An eligibility report  
- A structured submission checklist  
- A detailed report with actionable recommendations  

In the **Chatbot** tab, you can ask questions about the document and receive answers based on its content.
""")

uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        rfp_text = parse_pdf_streamlit(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        rfp_text = None

    if rfp_text:
        st.success("RFP loaded successfully!")
        st.subheader("RFP Preview (first 1000 characters)")
        st.text_area("RFP Preview", rfp_text[:1000], height=200)

        # Initialize tabs
        tabs = st.tabs(["Detailed Report", "Chatbot"])

        # Detailed Report Tab
        with tabs[0]:
            st.subheader("Generate Detailed Report")
            if st.button("Generate Report"):
                with st.spinner("Analyzing RFP... This may take a moment."):
                    # Initialize agents for detailed report generation
                    eligibility_agent = EligibilityAgent()
                    checklist_agent = SubmissionChecklistGenerator(rfp_text)
                    detailed_report_agent = DetailedReportAgent()
                    risk_agent = RiskAssessmentAgent() 
                    
                    # Execute analysis using the agents
                    eligibility_report = eligibility_agent.execute(rfp_text)
                    submission_checklist = checklist_agent.execute()
                    # If you want to include risk analysis, uncomment:
                    risk_report = risk_agent.execute(rfp_text)
                    
                    # Generate the formatted detailed Markdown report
                    markdown_report = detailed_report_agent.generate_formatted_report(
                        eligibility_report, submission_checklist,risk_report
                    )
                    
                    st.subheader("Aggregated Eligibility Report")
                    st.json(eligibility_report)
                    st.subheader("Submission Checklist")
                    st.json(submission_checklist)
                    st.subheader("Detailed Report (Markdown)")
                    st.markdown(markdown_report, unsafe_allow_html=True)
                    
                    # Store outputs in session state for later feedback and export
                    final_output = {
                        "eligibility_report": eligibility_report,
                        "submission_checklist": submission_checklist,
                        "risk_report":risk_report,
                        "detailed_report": markdown_report
                    }
                    st.session_state["final_output"] = final_output

            st.markdown("---")
            st.subheader("Provide Feedback on the Report")
            user_feedback = st.text_area("Your Feedback (e.g., flag false positives, missing criteria, suggestions):", height=150)
            if st.button("Submit Feedback"):
                if "final_output" in st.session_state:
                    st.session_state["final_output"]["feedback"] = user_feedback
                    st.success("Feedback submitted!")
                    st.subheader("Final Combined Report with Feedback")
                    st.json(st.session_state["final_output"])
                    with st.spinner("Uploading to DB"):
                        inserted_id = upload_to_mongodb(dict(st.session_state["final_output"]))
                else:
                    st.warning("Please generate the report first.")

        # Chatbot Tab
        with tabs[1]:
            st.subheader("Chat with the RFP Document")
            # Initialize the RAG retriever only once and store it in session state
            if "retriever" not in st.session_state:
                temp_pdf_path = "temp_uploaded.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                retriever = RAGretriver(temp_pdf_path)
                with st.spinner("Loading document and initializing vector store..."):
                    retriever.load_doc()
                    retriever.init_vectorstore()
                st.session_state["retriever"] = retriever
                st.success("Document loaded and vector store initialized.")
            else:
                st.success("Using previously initialized vector store.")
            
            user_query = st.text_input("Enter your question about the RFP:")
            if st.button("Get Answer"):
                if user_query:
                    with st.spinner("Retrieving answer..."):
                        answer = st.session_state["retriever"].RAG_Retrieve(user_query)
                        st.subheader("Chatbot Answer")
                        st.write(answer)
                else:
                    st.warning("Please enter a query.")
else:
    st.info("Please upload an RFP PDF to begin the analysis.")
