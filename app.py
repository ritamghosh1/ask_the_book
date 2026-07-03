import os
import tempfile
import streamlit as st  # type: ignore
from embed import VectorDBManager
from retriever import HybridRetriever
from llm import RAGGenerator

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Ask the Book", page_icon="📚", layout="centered")

st.title("📚 Ask the Book")
st.markdown("Upload any textbook, manual, or document and ask questions about it instantly.")

# --- SESSION STATE MANAGEMENT ---
# We use session state so the app doesn't forget the book/chat every time the user clicks a button
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- SIDEBAR: FILE UPLOAD  ---
with st.sidebar:
    st.header("1. Upload your Book")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # If a new file is uploaded, clear the old memory and process the new one
    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = [] # Clear old chat
        
        #  Progress Bar
        progress_bar = st.progress(10, text="Saving file temporarily...")
        
        # Save the uploaded file to a temporary location so our ingestion script can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            progress_bar.progress(30, text="Extracting and chunking text (this takes a moment)...")
            manager = VectorDBManager()
            index, nodes = manager.process_and_store(tmp_path)  # type: ignore
            
            progress_bar.progress(70, text="Building Hybrid Search & BM25 Indexes...")
            st.session_state.retriever = HybridRetriever(index, nodes)
            
            progress_bar.progress(90, text="Initializing Llama 3.1 LLM...")
            st.session_state.generator = RAGGenerator()
            
            progress_bar.progress(100, text="Ready!")
            st.success(f"Successfully processed {uploaded_file.name}!")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.retriever = None
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# --- CHAT INTERFACE ---
if st.session_state.retriever:
    
    # 1. Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            #  Show source chunks in an expander for assistant messages
            if "chunks" in msg and msg["chunks"]:
                with st.expander("View source chunks"):
                    for i, node in enumerate(msg["chunks"]):
                        page = node.metadata.get("page", "Unknown")
                        st.write(f"**Chunk {i+1} (Source: Page {page})**")
                        # Truncate to 500 characters so it doesn't clutter the screen
                        st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                        st.divider()

    # 2. Accept User Input
    if prompt := st.chat_input("Ask a question about the uploaded book..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 3. Generate Assistant Response
        with st.chat_message("assistant"):
            # Move the spinner so it only shows while retrieving, not while typing
            with st.spinner("Searching book..."):
                best_chunk_tuples = st.session_state.retriever.retrieve(prompt, top_k=3)
                just_nodes = [node for node, score in best_chunk_tuples]
                
            # Streaming LLM Response
            stream = st.session_state.generator.generate_stream(prompt, just_nodes) # type: ignore
            answer = st.write_stream(stream)
            
            # Display the source chunks in an expander
            with st.expander("View source chunks"):
                for i, node in enumerate(just_nodes):
                    page = node.metadata.get("page", "Unknown")
                    st.write(f"**Chunk {i+1} (Source: Page {page})**")
                    st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                    st.divider()
                        
        # Save the assistant's answer and chunks to the chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "chunks": just_nodes
        })
else:
    # If no book is uploaded yet, show a welcome message
    st.info("👈 Please upload a PDF in the sidebar to begin.")