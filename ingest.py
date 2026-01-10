import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "data/docs"
VECTORSTORE_PATH = "vectorstore"

def load_documents():
    documents = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def get_or_create_vectorstore(embeddings):
    
    docs = []

    for file in os.listdir("data/docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data/docs", file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def main():
    docs = load_documents()
    chunks = split_documents(docs)
    get_or_create_vectorstore(chunks)
    print(f"✅ Ingested {len(chunks)} chunks into FAISS")

if __name__ == "__main__":
    main()
