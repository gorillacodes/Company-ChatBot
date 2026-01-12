import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore"


def get_rag_chain():
    # ✅ SAME embeddings used during ingest
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    def answer(question: str) -> str:
        docs = retriever.invoke(question)

        context = "\n\n".join(
            f"[{d.metadata.get('book','Unknown')}] {d.page_content}"
            for d in docs
        )

        prompt = f"""
You are a Harry Potter expert assistant.

Use the provided context from the Harry Potter books to answer the question.
If the context contains multiple references, combine them into a clear answer.

If the answer is not clearly present, say:
"I don't find that explicitly stated in the books."

Context:
{context}

Question:
{question}

Answer (mention book name if relevant):
"""

        return llm.invoke(prompt).content

    return answer
