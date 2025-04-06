import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import PINECONE_KEY, OPENROUTER_KEY

os.environ['PINECONE_API_KEY'] = PINECONE_KEY
os.environ['OPENROUTER_API_KEY'] = OPENROUTER_KEY


try:
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    print("Embeddings initialized successfully!")
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    print("Openrouter initialized successfully!")
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    pinecone_index = pc.Index("consultaddhackathonchatbot")
    print("Pinecone initialized successfully!")
    
except Exception as e:
    print("Error initializing embeddings:", e)



index_name = "consultaddhackathonchatbot33"
namespace = "pdfdata1"

class RAGretriver:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None

    def load_doc(self):
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        self.doc = Document(page_content=text, metadata={"source": self.pdf_path})
        
    def tiktoken_len(self, text):
        import tiktoken
        tokenizer = tiktoken.get_encoding('p50k_base')
        return len(tokenizer.encode(text, disallowed_special=()))
    
    def init_vectorstore(self):
        if self.doc is None:
            self.load_doc()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=self.tiktoken_len
        )
        documents = splitter.split_documents([self.doc])
        
        # Delete the index if it exists
        try:
            pc.delete_index(index_name)
        except Exception as e:
            print("Index may not exist. Continuing...")

        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        vectorstore = PineconeVectorStore.from_texts(
            [d.page_content for d in documents],
            hf_embeddings,
            index_name=index_name,
            namespace=namespace
        )
        # After creation, reassign the index to pinecone_index variable
        global pinecone_index
        pinecone_index = pc.Index(index_name)
        return vectorstore

    def RAG_Retrieve(self, query):
        query_embedding = hf_embeddings.embed_query(query)
        top_matches = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )

        contexts = [match['metadata'].get("text", "") + "\n" + match.get("id", "") for match in top_matches['matches']]
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        primer = """You are a knowledgeable assistant. Answer questions using only the provided PDF content. Be precise, clear, and use facts from the document."""
        res = openrouter_client.chat.completions.create(
            model="mistralai/mistral-nemo",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
            ]
        )
        return res.choices[0].message.content
