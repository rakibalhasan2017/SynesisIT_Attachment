import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

# -------------------- SETUP --------------------
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# -------------------- STYLING --------------------
st.set_page_config(page_title="📚 RAG + Web Agent Chatbot", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background: #f4ecd8;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #2C3E50;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #7F8C8D;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .upload-box {
            border: 2px dashed #4A90E2;
            border-radius: 10px;
            padding: 1.5rem;
            background-color: #fefefe;
        }
        .user-msg {
            background-color: #d9eaff;
            padding: 0.8rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            color: #1B4F72;
        }
        .bot-msg {
            background-color: #f4f6f7;
            padding: 0.8rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            color: #2C3E50;
        }
        div.stButton > button:first-child {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            height: 2.5em;
            width: 100%;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #3b7fd4;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- BACKEND FUNCTIONS --------------------

def load_llm(huggingface_repo_id):
     llm = HuggingFaceEndpoint(
          repo_id=huggingface_repo_id,
            task="conversational",
            temperature=0.5,
             max_new_tokens=1000
     )
     return llm
     
CUSTOM_PROMPT_TEMPLATE = """
Use the information that has already been shared in the context to answer the user's question. 
If you're unsure of the answer, simply say that the information is not available in the PDF/PDFs 
instead of trying to create an answer.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

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

# -------------------- UI HEADER --------------------
st.markdown('<div class="title">🤖 RAG + Web Agent ChatBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask from uploaded PDFs or get answers from the web</div>', unsafe_allow_html=True)

# -------------------- UPLOAD SECTION --------------------
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("📤 Upload one or more PDF files", accept_multiple_files=True, type="pdf")
    st.markdown('</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = None

# -------------------- CREATE VECTORSTORE --------------------
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

# -------------------- WEB SEARCH AGENT --------------------
search_tool = DuckDuckGoSearchRun()

agent_llm = ChatGroq(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.0,
    groq_api_key=os.environ["GROQ_API_KEY"],
)

agent = initialize_agent(
    tools=[search_tool],
    llm=agent_llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------- CLEAR BUTTON --------------------
st.markdown("### 🧹 Manage Chat")
if st.button("Clear History"):
    st.session_state.messages = []
    st.toast("Chat history cleared!", icon="🧼")

# -------------------- CHAT DISPLAY --------------------
st.markdown("### 💬 Chat History")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 <b>Bot:</b> {msg["content"]}</div>', unsafe_allow_html=True)

# -------------------- CHAT INPUT --------------------
prompt = st.chat_input("💭 Ask something (PDF-based or general web question)...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-msg">👤 <b>You:</b> {prompt}</div>', unsafe_allow_html=True)

    with st.spinner("🤔 Thinking..."):
        # Case 1: PDFs are uploaded → Use RAG
        if qa_chain:
            response = qa_chain.invoke({'query': prompt})
            bot_reply = response["result"].strip() or "The information is not found in the PDF(s) you gave."
        # Case 2: No PDFs → Use DuckDuckGo Agent
        else:
            bot_reply = agent.run(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.markdown(f'<div class="bot-msg">🤖 <b>Bot:</b> {bot_reply}</div>', unsafe_allow_html=True)
