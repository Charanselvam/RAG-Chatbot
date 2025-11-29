# app.py
import streamlit as st
from rag_bot import build_vectorstore, create_rag_chain, ask

st.title("Local RAG Bot (LangChain + Ollama)")

if st.button("Ingest Documents"):
    with st.spinner("Processing PDFs..."):
        build_vectorstore()
        st.session_state.chain = create_rag_chain()
    st.success("Ready!")

query = st.text_input("Ask about your documents:")
if query and "chain" in st.session_state:
    with st.spinner("Thinking..."):
        answer, sources = ask(st.session_state.chain, query)
    st.markdown("**Answer:**")
    st.write(answer)
    if sources:
        st.markdown(f"**Sources:** {', '.join(sources)}")