import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate



load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


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


