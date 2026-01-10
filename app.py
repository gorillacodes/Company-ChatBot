import streamlit as st
from rag_chain import get_rag_chain
import subprocess


st.set_page_config(page_title="Leave Policy Bot", layout="wide")
st.title("📘 CDC Leave Policy Bot")

if st.button("🔄 Rebuild Knowledge Base"):
    with st.spinner("Rebuilding vector store..."):
        subprocess.run(["python", "ingest.py"])
    st.success("Knowledge base rebuilt successfully.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask your queries...")

if not query:
    st.stop()
if query:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke({"query": query})

    answer = response["result"]
    sources = response["source_documents"]

    st.session_state.chat_history.append(
        {"question": query, "answer": answer, "sources": sources}
    )

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        with st.expander("Sources"):
            for doc in chat["sources"]:
                st.markdown(
                    f"- **{doc.metadata.get('source', 'Document')}**, page {doc.metadata.get('page', '')}"
                )
