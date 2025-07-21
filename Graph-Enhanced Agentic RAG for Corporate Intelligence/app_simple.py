"""Simplified working version of the Graph-Enhanced Agentic RAG system."""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
from src.document_processor import DocumentProcessor
from src.knowledge_extractor import KnowledgeExtractor
from src.simple_vector_store import VectorStore
import config

st.set_page_config(
    page_title="Graph-Enhanced Corporate Intelligence",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = {'entities': [], 'relations': []}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def initialize_system():
    """Initialize the core system components."""
    try:
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = DocumentProcessor()
        if 'knowledge_extractor' not in st.session_state:
            st.session_state.knowledge_extractor = KnowledgeExtractor()
        if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return False

def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file."""
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document..."):
            # Process document
            try:
                doc_result = st.session_state.doc_processor.process_document(temp_path)
                chunks = doc_result['chunks']
                
                if not chunks:
                    st.error("No text could be extracted from the document.")
                    return False
                    
            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")
                return False
            
            # Extract knowledge with error handling
            try:
                entities, relations = st.session_state.knowledge_extractor.process_document_chunks(chunks)
                
                # Ensure we have some entities
                if not entities:
                    st.warning("No entities were extracted. The document may not contain structured business information.")
                    # Create a minimal entity from filename
                    from src.knowledge_extractor import Entity
                    fallback_entity = Entity(
                        name=uploaded_file.name.replace('.pdf', ''),
                        type='document',
                        context=f"Document: {uploaded_file.name}",
                        confidence=0.5
                    )
                    entities = [fallback_entity]
                    
            except Exception as e:
                st.error(f"Error extracting knowledge: {e}")
                # Create fallback entities
                from src.knowledge_extractor import Entity
                fallback_entity = Entity(
                    name=uploaded_file.name.replace('.pdf', ''),
                    type='document',
                    context=f"Document: {uploaded_file.name}",
                    confidence=0.3
                )
                entities = [fallback_entity]
                relations = []
            
            # Add to knowledge base
            st.session_state.knowledge_base['entities'].extend(entities)
            st.session_state.knowledge_base['relations'].extend(relations)
            
            # Add to vector store
            try:
                chunk_texts = [chunk['text'] for chunk in chunks]
                chunk_metadata = [{'source': uploaded_file.name, 'chunk_id': i} for i in range(len(chunks))]
                st.session_state.vector_store.add_documents(chunk_texts, chunk_metadata)
            except Exception as e:
                st.error(f"Error adding to vector store: {e}")
                return False
            
            # Update processed documents
            st.session_state.processed_documents.append({
                'name': uploaded_file.name,
                'chunks': len(chunks),
                'entities': len(entities),
                'relations': len(relations)
            })
        
        return True
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return False
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def answer_query(query):
    """Answer a user query using the knowledge base."""
    try:
        if st.session_state.vector_store is None or st.session_state.vector_store.index is None:
            return "No documents have been processed yet. Please upload documents first."
        
        # Search vector store
        results = st.session_state.vector_store.search(query, k=3)
        
        if not results:
            return "I don't have sufficient information to answer your query. Please upload relevant documents."
        
        # Create answer from top results
        context_chunks = []
        for result in results:
            context_chunks.append(result['text'][:500])  # First 500 chars
        
        context = " ".join(context_chunks)
        
        # Simple answer generation (in production, you'd use an LLM here)
        answer = f"Based on the available documents:\\n\\n"
        answer += f"**Context:** {context[:800]}...\\n\\n"
        answer += f"**Key Information:**\\n"
        
        # Add relevant entities
        relevant_entities = [e for e in st.session_state.knowledge_base['entities'] 
                           if any(word in e.name.lower() for word in query.lower().split())]
        
        if relevant_entities:
            answer += f"• **Entities mentioned:** "
            answer += ", ".join([f"{e.name} ({e.type})" for e in relevant_entities[:5]])
            answer += "\\n"
        
        # Add relevant relations
        relevant_relations = [r for r in st.session_state.knowledge_base['relations']
                            if any(word in r.subject.lower() or word in r.object.lower() 
                                 for word in query.lower().split())]
        
        if relevant_relations:
            answer += f"• **Relationships:** "
            for rel in relevant_relations[:3]:
                answer += f"{rel.subject} → {rel.predicate} → {rel.object}; "
            answer += "\\n"
        
        answer += f"\\n**Sources:** {len(results)} document chunks"
        
        return answer
        
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        return f"Error processing query: {e}"

# Main UI
st.title("🧠 Graph-Enhanced Corporate Intelligence")
st.markdown("---")

# Initialize system
if not initialize_system():
    st.stop()

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [doc['name'] for doc in st.session_state.processed_documents]:
                if st.button(f"Process {uploaded_file.name}"):
                    if process_uploaded_file(uploaded_file):
                        st.success(f"✅ Processed {uploaded_file.name}")
                        st.rerun()
    
    # Show processed documents
    if st.session_state.processed_documents:
        st.subheader("Processed Documents")
        for doc in st.session_state.processed_documents:
            st.write(f"📄 **{doc['name']}**")
            st.write(f"  • {doc['chunks']} chunks")
            st.write(f"  • {doc['entities']} entities")
            st.write(f"  • {doc['relations']} relations")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Query Interface")
    
    # Query input
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What challenges did Microsoft face in 2023?"
    )
    
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Processing query..."):
                answer = answer_query(query)
            
            st.subheader("Answer:")
            st.markdown(answer)
        else:
            st.warning("Please enter a query.")
    
    # Sample queries
    st.subheader("💡 Sample Queries")
    sample_queries = [
        "What are the main risk factors?",
        "Who are the key executives?",
        "What was the revenue growth?",
        "What partnerships were announced?",
        "What are the competitive advantages?"
    ]
    
    for sample_query in sample_queries:
        if st.button(sample_query):
            with st.spinner("Processing query..."):
                answer = answer_query(sample_query)
            
            st.subheader("Answer:")
            st.markdown(answer)

with col2:
    st.header("📊 System Status")
    
    # System metrics
    total_docs = len(st.session_state.processed_documents)
    total_entities = len(st.session_state.knowledge_base['entities'])
    total_relations = len(st.session_state.knowledge_base['relations'])
    
    st.metric("Documents Processed", total_docs)
    st.metric("Entities Extracted", total_entities)
    st.metric("Relations Found", total_relations)
    
    # Knowledge base preview
    if st.session_state.knowledge_base['entities']:
        st.subheader("🏷️ Recent Entities")
        for entity in st.session_state.knowledge_base['entities'][-5:]:
            st.write(f"• {entity.name} ({entity.type})")
    
    if st.session_state.knowledge_base['relations']:
        st.subheader("🔗 Recent Relations")
        for relation in st.session_state.knowledge_base['relations'][-3:]:
            st.write(f"• {relation.subject} → {relation.predicate} → {relation.object}")

# Footer
st.markdown("---")
st.markdown("**Graph-Enhanced Agentic RAG for Corporate Intelligence** | Built with Streamlit")
