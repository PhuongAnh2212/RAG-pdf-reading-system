import streamlit as st
import os
from process_pdf import process_pdf
from rag_pipeline import chunk_text, create_vector_store, query_rag

if not os.path.exists("data"):
    os.makedirs("data")

st.title("RAG PDF Query System")

if "vector_stores" not in st.session_state:
    st.session_state["vector_stores"] = {}

# PDF Upload and Processing with Progress Bar
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        pdf_path = os.path.join("data", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})...")
        text = process_pdf(pdf_path)
        chunks = chunk_text(text)
        vector_store = create_vector_store(chunks)
        
        st.session_state["vector_stores"][uploaded_file.name] = vector_store
        progress = int(((i + 1) / total_files) * 100)
        progress_bar.progress(progress)
    
    status_text.text(f"All {total_files} PDFs processed and ready for queries!")
    progress_bar.empty()

# Display uploaded PDFs
if st.session_state["vector_stores"]:
    st.write("**Uploaded PDFs:**", list(st.session_state["vector_stores"].keys()))

# Query Input
query = st.text_input("Ask a question about the PDFs:")
if query and st.session_state["vector_stores"]:
    with st.spinner("Generating answer..."):
        response = query_rag(st.session_state["vector_stores"], query)
    st.write("**Answer:**", response)