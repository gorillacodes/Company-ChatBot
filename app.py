import streamlit as st
from rag_chain import get_rag_chain

st.set_page_config(page_title="Harry Potter RAG Bot", layout="centered")

st.title("🪄 Harry Potter Knowledge Bot")
st.caption("Ask questions across all 7 Harry Potter books")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Chat input
query = st.chat_input("Ask anything about Harry Potter...")

if query:
    with st.spinner("Consulting the books..."):
        response = st.session_state.qa_chain.invoke({"query": query})

        answer = response["result"]
        sources = response["source_documents"]

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer,
                "sources": sources,
            }
        )


# Render chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        if chat["sources"]:
            with st.expander("Sources"):
                for doc in chat["sources"]:
                    st.markdown(
                        f"- **{doc.metadata.get('book', 'Unknown Book')}**, page {doc.metadata.get('page', '?')}"
                    )
