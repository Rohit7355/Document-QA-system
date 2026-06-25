# """
# Streamlit UI for RAG system.

# Run with: streamlit run app.py
# """

import streamlit as st
import os
import time
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="📚 RAG Q&A System", page_icon="🤖", layout="wide")
st.title("📚 RAG Q&A System")
st.markdown("*Ask questions about your documents*")

@st.cache_resource 
def get_rag_pipeline():
    return RAGPipeline()

rag = get_rag_pipeline()

with st.sidebar:
    st.header("⚙️ Setup")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ GROQ_API_KEY not set!")
    else:
        st.success("✓ API Key configured")

    st.divider()
    st.header("📤 Add Documents")

    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md"])
    if uploaded_file:
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("📑 Index Document"):
            with st.spinner("Indexing..."):
                result = rag.index_document(temp_path)
            os.remove(temp_path)
            if result["status"] == "success":
                doc_name = result["file"]
                st.session_state.uploaded_documents[doc_name] = {
                    "chunks": result["chunks_created"],
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success(f"✓ Indexed {result['chunks_created']} chunks from {doc_name}")
            else:
                st.error(f"Error: {result['message']}")

    st.divider()
    st.header("📋 Your Documents")
    stats = rag.get_stats()
    st.metric("Total Chunks", stats["total_chunks"])
    if st.session_state.uploaded_documents:
        st.subheader("Indexed Documents:")
        for doc_name, info in st.session_state.uploaded_documents.items():
            st.write(f"📄 {doc_name} — {info['chunks']} chunks")


    st.divider()
    st.header("🗑️ Document Management")

    if st.session_state.uploaded_documents:
       st.subheader("Delete Individual Documents:")
    
     # Create delete buttons for each document
       for doc_name in st.session_state.uploaded_documents.keys():
         col1, col2 = st.columns([3, 1])
        
         with col1:
            st.write(f"📄 {doc_name}")
        
         with col2:
            if st.button("🗑️ Delete", key=f"delete_{doc_name}"):
                # Delete from ChromaDB
                result = rag.delete_document(doc_name)
                
                if result["status"] == "success":
                    # Remove from session state
                    del st.session_state.uploaded_documents[doc_name]
                    st.success(f"✓ Deleted {doc_name}")
                    st.rerun()
                else:
                    st.error(f"Error: {result['message']}")
    
       st.divider()
    
     # Clear all button
       if st.button("⚠️ Clear ALL Documents", use_container_width=True):
        rag.clear()
        st.session_state.uploaded_documents = {}
        st.success("✓ All documents cleared!")
        st.rerun()
    else:
         st.info("No documents uploaded yet")

# ---------------- MAIN AREA ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("❓ Ask a Question")

    if st.session_state.uploaded_documents:
        doc_options = ["All Documents"] + list(st.session_state.uploaded_documents.keys())
        selected_doc = st.selectbox("Search in:", doc_options)
    else:
        selected_doc = None
        st.info("📄 Upload a document first to ask questions")

    question = st.text_area("Enter your question:", placeholder="E.g., What is deep learning?", height=80)

    if st.button("🔍 Search & Answer", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question")
        elif not st.session_state.uploaded_documents:
            st.warning("Please upload a document first")
        else:
            with st.spinner("Searching and generating answer..."):
                source_filter = None if selected_doc == "All Documents" else selected_doc
                result = rag.query(question, source_document=source_filter)

            st.divider()
            st.subheader("💡 Answer")
            st.markdown(result["answer"])

            st.subheader(f"📍 Retrieved Context ({len(result['retrieved_chunks'])} chunks)")
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                with st.expander(f"Chunk {i} (similarity: {chunk['similarity']:.2%})"):
                    st.markdown(chunk['text'])

            if selected_doc and selected_doc != "All Documents":
                st.info(f"✓ Searched in: {selected_doc}")
            else:
                st.info("✓ Searched in all documents")

with col2:
    st.header("ℹ️ About")
    st.markdown("""
    **How to use:**
    1. Upload documents (PDF/TXT)
    2. Select which document to search
    3. Ask questions
    4. Get answers from THAT document

    **Features:**
    - Multiple documents support
    - Document-specific search
    - Shows retrieved chunks
    - Shows similarity scores
    """)































# """
# Streamlit UI for RAG system.
# Simple, functional student-level interface.

# Run with: streamlit run app.py
# """

# import streamlit as st
# import os
# from pathlib import Path
# from rag_pipeline import RAGPipeline

# # Page config
# st.set_page_config(
#     page_title="📚 RAG Q&A System",
#     page_icon="🤖",
#     layout="wide"
# )

# st.title("📚 RAG Q&A System")
# st.markdown("*Ask questions about your documents*")

# # Initialize pipeline (cached so it doesn't reload every interaction)
# @st.cache_resource
# def get_rag_pipeline():
#     """Initialize RAG pipeline once and cache it."""
#     return RAGPipeline()

# rag = get_rag_pipeline()

# # Sidebar
# with st.sidebar:
#     st.header("⚙️ Setup")
    
#     # API Key check
#     if not os.getenv("GROQ_API_KEY"):
#         st.warning("⚠️ GROQ_API_KEY not set!")
#         st.markdown("""
#         1. Get free API key: https://console.groq.com/
#         2. Set environment variable:
#            ```bash
#            export GROQ_API_KEY="your-key-here"
#            ```
#         """)
#     else:
#         st.success("✓ API Key configured")
    
#     st.divider()
    
#     # Document upload
#     st.header("📤 Add Documents")
#     uploaded_file = st.file_uploader(
#         "Upload PDF or TXT",
#         type=["pdf", "txt", "md"],
#         help="PDF or text files only"
#     )
    
#     if uploaded_file:
#         # Save uploaded file temporarily
#         temp_path = f"./temp_{uploaded_file.name}"
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         if st.button("📑 Index Document"):
#             with st.spinner("Indexing..."):
#                 result = rag.index_document(temp_path)
            
#             # Cleanup
#             os.remove(temp_path)
            
#             if result["status"] == "success":
#                 st.success(f"✓ Indexed {result['chunks_created']} chunks")
#             else:
#                 st.error(f"Error: {result['message']}")
    
#     st.divider()
    
#     # Stats
#     st.header("📊 Stats")
#     stats = rag.get_stats()
#     st.metric("Total Chunks Indexed", stats["total_chunks"])
#     st.caption(f"Model: {stats['model']}")


# # Main area
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("❓ Ask a Question")
    
#     # Query input
#     question = st.text_area(
#         "Enter your question:",
#         placeholder="E.g., What is deep learning?",
#         height=80
#     )
    
#     # Submit button
#     if st.button("🔍 Search & Answer", use_container_width=True):
#         if not question.strip():
#             st.warning("Please enter a question")
#         elif stats["total_chunks"] == 0:
#             st.warning("Please upload and index a document first")
#         else:
#             with st.spinner("Retrieving context and generating answer..."):
#                 result = rag.query(question)
            
#             st.divider()
            
#             # Answer
#             st.subheader("Answer")
#             st.markdown(result["answer"])
            
#             # Retrieved chunks
#             st.subheader(f"📍 Retrieved Context ({len(result['retrieved_chunks'])} chunks)")
            
#             for i, chunk in enumerate(result["retrieved_chunks"], 1):
#                 with st.expander(f"Chunk {i} (similarity: {chunk['similarity']:.2%})"):
#                     st.markdown(chunk['text'])

# # with col2:
# #     st.header("ℹ️ About")
# #     st.markdown("""
# #     **What is RAG?**
    
# #     Retrieval-Augmented Generation means:
# #     1. Find relevant chunks from your docs
# #     2. Send them as context to an LLM
# #     3. Get grounded answers
    
# #     **Why use it?**
# #     - Answers based on YOUR data
# #     - No hallucinations
# #     - Cost-effective
# #     - Works for any document
    
# #     **How it works:**
# #     1. Upload documents
# #     2. System creates embeddings
# #     3. Stores in vector database
# #     4. Retrieves relevant parts
# #     5. LLM generates answer
# #     """)
    
# #     st.divider()
    
# #     st.markdown("""
# #     **Stack:**
# #     - 🔗 Embeddings: sentence-transformers
# #     - 📦 Vector DB: ChromaDB
# #     - 🧠 LLM: Groq API
# #     - 🎨 UI: Streamlit
# #     """)

# # # Footer
# # st.divider()
# # st.caption("📚 A student project for learning RAG systems | Built with ❤️")
