import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils import extract_text_from_pdf, get_vectorstore_from_text
import os
import tempfile

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📄 Chat With Your Report")
st.caption("Upload a PDF and ask questions!")

# Upload and process the PDF
uploaded_file = st.file_uploader("Upload your PDF report", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF Text Extracted")

    with st.spinner("🔍 Generating embeddings..."):
        vectorstore = get_vectorstore_from_text(text)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Chat interface
    query = st.text_input("Ask a question about the report:")

    if query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain.run(query)
        st.markdown(f"**Answer:** {result}")
