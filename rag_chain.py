from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
from ingest import get_or_create_vectorstore

load_dotenv()

VECTORSTORE_PATH = "vectorstore"

def get_rag_chain():
    embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = get_or_create_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(search_type = "similarity",search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key = st.secrets["GROQ_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Harry Potter expert assistant.

Answer the question strictly using the provided context from the Harry Potter books.
If the answer is not present in the context, say:
"I don't find that explicitly stated in the books."
If multiple books are referenced in the context, base your answer on the most relevant one.

Context:
{context}

Question:
{question}

Answer:
- Be concise and factual.
- Mention the book name if it is identifiable from the context.
"""
)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
