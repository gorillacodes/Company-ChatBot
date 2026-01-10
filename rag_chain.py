from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

VECTORSTORE_PATH = "vectorstore"

def get_rag_chain():
    embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key = st.secrets["GROQ_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant answering questions using ONLY the provided context.
If the answer is not present, say: "I don’t have that information."

Context:
{context}

Question:
{question}
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
