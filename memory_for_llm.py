from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH='data/'
# Load the PDF file
def load_pdf(file_path):
    loader= DirectoryLoader(file_path, glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

docs = load_pdf(file_path=DATA_PATH)
# print(f"Loaded {len(docs)}")


# Creating chunks of text

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

chunks = create_chunks(extracted_data=docs)
# print(f"Created {len(chunks)} chunks of text")

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# store in FAISS
DB_FAISS_PATH='vectorstore/faiss_db'
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)