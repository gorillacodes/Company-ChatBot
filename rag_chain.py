import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

VECTORSTORE_PATH = "vectorstore"


def get_rag_chain():
    # Load FAISS index (prebuilt locally)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings=None,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    def answer_question(question: str) -> str:
        # 1. Retrieve documents
        docs = retriever.invoke(question)

        # 2. Build context manually
        context = "\n\n".join(
            f"[{d.metadata.get('book', 'Unknown')}] {d.page_content}"
            for d in docs
        )

        # 3. Build prompt
        prompt = f"""
You are a Harry Potter expert assistant.

Answer the question strictly using the context from the Harry Potter books.
If the answer is not present, say:
"I don't find that explicitly stated in the books."

Context:
{context}

Question:
{question}

Answer (include book name if possible):
"""

        # 4. Call LLM
        response = llm.invoke(prompt)
        return response.content

    return answer_question
