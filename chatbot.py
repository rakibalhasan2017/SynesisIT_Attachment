import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"




def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the information that has already been shared in the context to answer the user's question. If you're unsure of the answer, simply say that the information is not available in the PDF/PDFs instead of trying to create an answer.
Don't give any information that's not in the context provided.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Extract text from multiple PDFs and chunk all together
def extract_and_chunk_multiple_pdfs(pdf_files, chunk_size=1000, chunk_overlap=200):
    combined_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(combined_text)
    return chunks

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
if os.path.exists(DB_FAISS_PATH):
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    db = None

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Multiple PDFs uploader
uploaded_files = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type="pdf")

qa_chain = None  # initialize chain




if uploaded_files:
    chunks = extract_and_chunk_multiple_pdfs(uploaded_files)
    new_db = FAISS.from_texts(chunks, embedding_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            groq_api_key=os.environ["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=new_db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )


# Clear chat history button
if st.button("Clear History", type="primary"):
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about the PDF(s):")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({'query': prompt})
        bot_reply = response["result"].strip() or "The information is not found in the PDF(s) you gave."
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
