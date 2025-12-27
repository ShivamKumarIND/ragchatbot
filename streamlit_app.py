"""
Streamlit Web UI for RAG Chatbot
Interactive web interface for document upload and chat
"""
import streamlit as st
import os
import shutil
from pathlib import Path
from datetime import datetime

from document_processor import DocumentProcessor
from vector_store import get_vector_store
from rag_chain import get_chatbot
from llm_loader import get_llm_loader

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-doc {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        try:
            st.session_state.chatbot = get_chatbot()
            st.session_state.vector_store = get_vector_store()
            st.session_state.document_processor = DocumentProcessor()
            st.session_state.llm_loader = get_llm_loader()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'docx', 'doc', 'html', 'htm', 'csv', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, Word, HTML, CSV, Excel, or Text files"
    )
    
    if uploaded_files:
        if st.button("📤 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    processed_count = 0
                    total_chunks = 0
                    
                    for uploaded_file in uploaded_files:
                        # Save file
                        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        split_docs = st.session_state.document_processor.process_document(file_path)
                        
                        # Add to vector store
                        st.session_state.vector_store.add_documents(split_docs)
                        
                        processed_count += 1
                        total_chunks += len(split_docs)
                    
                    # Reinitialize chatbot
                    st.session_state.chatbot.reinitialize_chain()
                    
                    st.success(f"✅ Processed {processed_count} file(s) into {total_chunks} chunks")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚙️ System Status")
    
    try:
        doc_count = st.session_state.vector_store.get_collection_count()
        st.metric("Documents Indexed", doc_count)
        
        current_llm = st.session_state.llm_loader.config.get('managerLLM')
        st.info(f"🤖 LLM: {current_llm}")
        
        supported_formats = st.session_state.document_processor.get_supported_extensions()
        st.caption(f"📄 Formats: {', '.join(supported_formats[:4])}...")
        
    except Exception as e:
        st.error(f"Status error: {str(e)}")
    
    st.markdown("---")
    
    # Actions
    st.markdown("### 🛠️ Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.clear_memory()
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset DB", use_container_width=True):
            try:
                st.session_state.vector_store.delete_collection()
                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR)
                    os.makedirs(UPLOAD_DIR)
                st.success("Database reset!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("RAG-based AI chatbot using LangChain and Groq LLM")
    st.caption("Ask questions about your uploaded documents")

# Main content
st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)

if not st.session_state.initialized:
    st.error("⚠️ System not initialized. Please check your configuration.")
    st.stop()

# Check if documents are uploaded
doc_count = st.session_state.vector_store.get_collection_count()
if doc_count == 0:
    st.info("👈 Upload documents using the sidebar to get started!")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                        st.caption(source['content'][:150] + "...")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if doc_count == 0:
        st.warning("⚠️ Please upload documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get response from chatbot
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    answer = response["answer"]
                    sources = response.get("source_documents", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                                st.caption(doc.page_content[:150] + "...")
                    
                    # Format sources for storage
                    formatted_sources = []
                    for doc in sources:
                        formatted_sources.append({
                            "content": doc.page_content[:200],
                            "metadata": doc.metadata
                        })
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": formatted_sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💬 Chat History: " + str(len(st.session_state.messages)))
with col2:
    st.caption("📄 Documents: " + str(doc_count))
with col3:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
