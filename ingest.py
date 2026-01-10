import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


DOCS_PATH = "data/docs"


def load_documents():
    documents = []
    for file in sorted(os.listdir(DOCS_PATH)):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            loaded_docs = loader.load()

            book_name = file.replace(".pdf", "").replace("_","")
            for d in loaded_docs:
                d.metadata["book"] = book_name
                d.metadata["source"] = book_name
            documents.extend(load_documents)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
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
