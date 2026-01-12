import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

VECTORSTORE_PATH = "vectorstore"


def get_rag_chain():
    # Load FAISS (prebuilt locally)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings=None,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # 🔑 Explicitly call retriever (prevents NoneType callable error)
    retrieve_docs = RunnableLambda(
        lambda question: retriever.get_relevant_documents(question)
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Harry Potter expert assistant.

Answer the question strictly using the provided context from the Harry Potter books.
If the answer is not present in the context, say:
"I don't find that explicitly stated in the books."

Context:
{context}

Question:
{question}

Answer (include book name if possible):
"""
    )

    rag_chain = (
        {
            "context": retrieve_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain
