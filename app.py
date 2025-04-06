import streamlit as st
import os
import json
import pymongo
from utils.fileparser import parse_pdf_streamlit
from agents.eligiblity import EligibilityAgent
from agents.checklist import SubmissionChecklistGenerator
from agents.report_agent import DetailedReportAgent
from agents.risk_assessment import RiskAssessmentAgent
from utils.RAGretriver import RAGretriver
import torch
from agents.proposal_writer import ProposalWriterAgent
from utils.text_utils import markdown_to_docx 
torch.classes.__path__ = []

# MongoDB uploader
def upload_to_mongodb(data):
    try:
        connection_string = "mongodb+srv://apurva:apurva123@cluster0.0nmuvte.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = pymongo.MongoClient(connection_string)
        db = client["consultadd"]
        collection = db["hackathon"]
        result = collection.insert_one(data)
        return result.inserted_id
    except Exception as e:
        st.error(f"Error uploading to MongoDB: {e}")
        return None

# Page setup
st.set_page_config(page_title="📄 RFP Analysis & Chatbot", layout="wide")

# Custom styling
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
        }
        .report-box {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .chatbot-box {
            background-color: #f2f0fc;
            border-left: 5px solid #6a0dad;
            padding: 1rem;
            border-radius: 10px;
        }
        .feedback-box {
            background-color: #fdf4ff;
            padding: 1rem;
            border-radius: 10px;
        }
       
        .report-box {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            color: #222222; /* <-- fix white text issue */
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar for chatbot
st.sidebar.header("💬 Chat with your RFP")
st.sidebar.markdown("Ask specific questions about the RFP document after uploading it.")
chat_query = st.sidebar.text_input("Your Question:")
if st.sidebar.button("Get Answer"):
    if "retriever" in st.session_state and chat_query:
        with st.spinner("Retrieving answer..."):
            answer = st.session_state["retriever"].RAG_Retrieve(chat_query)
            st.sidebar.success("Answer:")
            st.sidebar.write(answer)
    else:
        st.sidebar.warning("Please upload an RFP and initialize the chatbot.")

# Main header
st.title("📄 RFP Analysis & Detailed Insights")
st.markdown("Upload an **RFP PDF**, and let our system generate a full breakdown:")

# File upload
uploaded_file = st.file_uploader("📤 Upload your RFP (PDF Only)", type=["pdf"])

if uploaded_file is not None:
    try:
        rfp_text = parse_pdf_streamlit(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        rfp_text = None

    if rfp_text:
        st.success("✅ RFP loaded successfully!")
        st.subheader("📑 RFP Preview")
        st.text_area("Preview (first 1000 characters)", rfp_text[:1000], height=200)

        tab1, tab2, tab3 = st.tabs(["📋 Detailed Report", "📝 Feedback & Submission", "🧾 Proposal Document"])

        with tab1:
            st.subheader("🔍 Analyze the RFP")
            if st.button("Generate Report"):
                with st.spinner("Analyzing the RFP..."):
                    eligibility_agent = EligibilityAgent()
                    checklist_agent = SubmissionChecklistGenerator(rfp_text)
                    detailed_report_agent = DetailedReportAgent()
                    risk_agent = RiskAssessmentAgent()
                    
                    eligibility_report = eligibility_agent.execute(rfp_text)
                    submission_checklist = checklist_agent.execute()
                    risk_report = risk_agent.execute(rfp_text)

                    markdown_report = detailed_report_agent.generate_formatted_report(
                        eligibility_report, submission_checklist, risk_report
                    )

                    proposal_agent = ProposalWriterAgent()
                    generated_proposal = proposal_agent.generate_proposal(
                        eligibility_report,
                        submission_checklist,
                        risk_report,
                        markdown_report
                    )

                    final_output = {
                        "eligibility_report": eligibility_report,
                        "submission_checklist": submission_checklist,
                        "risk_report": risk_report,
                        "detailed_report": markdown_report,
                        "proposal":generated_proposal
                    }
                    st.session_state["final_output"] = final_output

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### ✅ Aggregated Eligibility")
                        st.json(eligibility_report)
                        st.markdown("#### 📌 Submission Checklist")
                        st.json(submission_checklist)

                    with col2:
                        st.markdown("#### 🧾 Complete Report")
                        st.markdown(f"<div class='report-box'>{markdown_report}</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("📬 Provide Feedback & Save Report")
            feedback = st.text_area("Your Feedback", placeholder="Flag issues or suggest improvements...", height=150)
            if st.button("Submit Feedback"):
                if "final_output" in st.session_state:
                    st.session_state["final_output"]["feedback"] = feedback
                    st.success("✅ Feedback submitted!")

                    with st.spinner("Uploading to database..."):
                        inserted_id = upload_to_mongodb(dict(st.session_state["final_output"]))
                        if inserted_id:
                            st.success(f"Report uploaded to database with ID: {inserted_id}")
                        else:
                            st.error("Something went wrong uploading to MongoDB.")
                    st.markdown("### 🧾 Final Report (with Feedback)")
                    st.json(st.session_state["final_output"])
                else:
                    st.warning("Please generate the report before submitting feedback.")
        

        with tab3:
            st.subheader("📄 Generated Proposal")
            if "final_output" in st.session_state:  
                st.markdown(st.session_state["final_output"]["proposal"], unsafe_allow_html=True)

            # Export to DOCX and offer download
                docx_path = markdown_to_docx(st.session_state["final_output"]["proposal"])
                with open(docx_path, "rb") as f:
                    st.download_button("📥 Download Proposal as Word Document", f, file_name="Generated_Proposal.docx")

        # Initialize chatbot retriever after analysis
        if "retriever" not in st.session_state:
            temp_path = "temp_uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            retriever = RAGretriver(temp_path)
            with st.spinner("Setting up chatbot..."):
                retriever.load_doc()
                retriever.init_vectorstore()
            st.session_state["retriever"] = retriever
else:
    st.info("📂 Please upload an RFP file to get started.")
