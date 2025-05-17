import os
import requests
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Get and print HF token
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HF_TOKEN")

# Optional: Set it explicitly in environment in case needed
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Step 2: Login to Hugging Face
try:
    login(token=HUGGINGFACEHUB_API_TOKEN)

except Exception as e:
    print("[ERROR] HuggingFace login failed:", e)

# Step 3: Define Endpoint URL (correct format)
ENDPOINT_URL = "mistralai/Mistral-7B-Instruct-v0.3"

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# Step 4: Test if the model is accessible via inference API
def test_model_api():
    print("[DEBUG] Testing model endpoint connectivity...")
    api_url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_REPO_ID}"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    try:
        test_response = requests.post(api_url, headers=headers, json={"inputs": "Hello"})
        print("[DEBUG] Inference API test response:", test_response.status_code)
        if test_response.status_code != 200:
            print("[WARNING] Model is not accessible or loading. Response:", test_response.json())
    except Exception as e:
        print("[ERROR] API test failed:", e)


# Step 5: Load the LLM
def load_llm(huggingface_repo_id):
    
    try:
        endpoint_url = f"https://api-inference.huggingface.co/models/{huggingface_repo_id}"
        llm = HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            temperature=0.7,
            max_new_tokens=512,
            task="conversational",
        )

        return llm
    except Exception as e:
        print("[ERROR] Failed to load LLM:", e)


# Step 6: Set custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 7: Load FAISS vector store
DB_FAISS_PATH = "vectorstore/faiss_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    
except Exception as e:
    print("[ERROR] Failed to load FAISS DB:", e)

# Step 8: Create QA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    
except Exception as e:
    print("[ERROR] Failed to create QA chain:", e)

# Step 9: Get user query and run it
try:
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})  # ✅ Corrected key
    print("[RESULT]", response["result"])
except Exception as e:
    print("[ERROR] Failed during query execution:", e)

