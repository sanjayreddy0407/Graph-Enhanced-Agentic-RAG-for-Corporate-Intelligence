"""Main Streamlit application for Graph-Enhanced Agentic RAG system."""

import streamlit as st
import os
import time
import json
from typing import Dict, Any, Optional, List
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
from config import Config, config
from src.document_processor import DocumentProcessor
from src.knowledge_extractor import KnowledgeExtractor
from src.vector_store import VectorStore, VectorStoreManager
from src.graph_store import GraphStore
from src.agents.query_classifier import QueryClassifierAgent
from src.agents.vector_agent import VectorAgent
from src.agents.graph_agent import GraphAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.utils.visualization import GraphVisualizer, DataVisualizer

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'knowledge_extractor' not in st.session_state:
        st.session_state.knowledge_extractor = KnowledgeExtractor()
    
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(Config.VECTOR_DB_PATH)
    
    if 'graph_store' not in st.session_state:
        st.session_state.graph_store = GraphStore(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD
        )
    
    if 'query_classifier' not in st.session_state:
        st.session_state.query_classifier = QueryClassifierAgent()
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def display_header():
    """Display the main application header."""
    st.markdown('<div class="main-header">🧠 Graph-Enhanced Corporate Intelligence</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Intelligent Q&A System</strong> that combines vector-based semantic retrieval with 
        graph-based relational understanding for comprehensive corporate intelligence.
    </div>
    """, unsafe_allow_html=True)

def document_upload_section():
    """Handle document upload and processing."""
    st.markdown('<div class="section-header">📄 Document Upload & Processing</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Corporate Documents (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload annual reports, 10-Ks, 10-Qs, or other corporate documents"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Save uploaded file
                    file_path = os.path.join(Config.DOCUMENTS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Process document
                        processed_doc = st.session_state.document_processor.process_document(file_path)
                        
                        # Extract knowledge
                        entities, relations = st.session_state.knowledge_extractor.process_document_chunks(
                            processed_doc['chunks']
                        )
                        
                        # Store in vector database
                        vector_store = st.session_state.vector_store_manager.get_store('main')
                        vector_store.add_document_chunks(
                            processed_doc['chunks'],
                            processed_doc['metadata']
                        )
                        
                        # Store in graph database
                        st.session_state.graph_store.add_entities(entities)
                        st.session_state.graph_store.add_relations(relations)
                        
                        # Add to processed documents list
                        st.session_state.processed_documents.append({
                            'filename': uploaded_file.name,
                            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'entities_count': len(entities),
                            'relations_count': len(relations),
                            'chunks_count': len(processed_doc['chunks'])
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
                
                # Save vector store
                st.session_state.vector_store_manager.save_all_stores()
                
                status_text.text("All documents processed successfully!")
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
    
    # Display processed documents
    if st.session_state.processed_documents:
        st.markdown("### 📋 Processed Documents")
        for doc in st.session_state.processed_documents:
            with st.expander(f"📄 {doc['filename']} - {doc['processed_at']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entities Extracted", doc['entities_count'])
                with col2:
                    st.metric("Relations Found", doc['relations_count'])
                with col3:
                    st.metric("Text Chunks", doc['chunks_count'])

def query_interface():
    """Main query interface for asking questions."""
    st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)
    
    # Sample queries
    with st.expander("💡 Sample Queries"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Semantic Queries:**")
            semantic_queries = [
                "What challenges did Microsoft face in 2023?",
                "Summarize Amazon's financial performance",
                "What are the key risks mentioned in the report?"
            ]
            for query in semantic_queries:
                if st.button(query, key=f"semantic_{query[:20]}"):
                    st.session_state.current_query = query
        
        with col2:
            st.markdown("**Relational Queries:**")
            relational_queries = [
                "Which companies did Apple acquire?",
                "Who is the CEO of Microsoft?",
                "List all partnerships of Google"
            ]
            for query in relational_queries:
                if st.button(query, key=f"relational_{query[:20]}"):
                    st.session_state.current_query = query
        
        with col3:
            st.markdown("**Hybrid Queries:**")
            hybrid_queries = [
                "Summarize Apple's performance and key partnerships",
                "What did the CEO say about AI and acquisitions?",
                "Analyze financial health and supplier relationships"
            ]
            for query in hybrid_queries:
                if st.button(query, key=f"hybrid_{query[:20]}"):
                    st.session_state.current_query = query
    
    # Main query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="Ask about financial performance, acquisitions, leadership, partnerships, or any corporate intelligence topic..."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", disabled=not query.strip())
    with col2:
        show_reasoning = st.checkbox("Show Reasoning", value=True)
    with col3:
        show_sources = st.checkbox("Show Sources", value=True)
    
    if ask_button and query.strip():
        with st.spinner("Analyzing your question and searching for answers..."):
            # Clear previous query from session state
            if 'current_query' in st.session_state:
                del st.session_state.current_query
            
            # Process the query
            result = process_query(query)
            
            # Display results
            display_query_results(result, show_reasoning, show_sources)
            
            # Add to conversation history
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'result': result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })

def process_query(query: str) -> Dict[str, Any]:
    """Process a user query through the complete pipeline."""
    start_time = time.time()
    
    try:
        # Step 1: Classify query
        classification = st.session_state.query_classifier.classify_query(query)
        
        # Step 2: Retrieve based on classification
        vector_results = None
        graph_results = None
        
        vector_store = st.session_state.vector_store_manager.get_store('main')
        vector_agent = VectorAgent(vector_store)
        graph_agent = GraphAgent(st.session_state.graph_store)
        
        if classification.suggested_approaches['use_vector']:
            vector_results = vector_agent.retrieve(query)
        
        if classification.suggested_approaches['use_graph']:
            graph_results = graph_agent.retrieve(query)
        
        # Step 3: Synthesize answer
        synthesis_agent = SynthesisAgent()
        synthesis_result = synthesis_agent.synthesize_answer(
            query=query,
            vector_results=vector_results,
            graph_results=graph_results
        )
        
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'classification': classification,
            'vector_results': vector_results,
            'graph_results': graph_results,
            'synthesis': synthesis_result,
            'total_time': total_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            'query': query,
            'error': str(e),
            'success': False,
            'total_time': time.time() - start_time
        }

def display_query_results(result: Dict[str, Any], show_reasoning: bool = True, show_sources: bool = True):
    """Display the results of a query."""
    if not result.get('success', False):
        st.error(f"Error processing query: {result.get('error', 'Unknown error')}")
        return
    
    # Main answer
    st.markdown("### 🎯 Answer")
    answer = result['synthesis'].answer
    confidence = result['synthesis'].confidence
    
    # Display answer with confidence indicator
    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
    st.markdown(f"""
    <div class="success-box">
        <strong>Answer:</strong><br>
        {answer}
        <br><br>
        <strong>Confidence:</strong> 
        <span style="color: {confidence_color}">
            {confidence:.1%} {'🟢' if confidence > 0.8 else '🟡' if confidence > 0.6 else '🔴'}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Query classification info
    classification = result['classification']
    st.markdown(f"""
    <div class="info-box">
        <strong>Query Type:</strong> {classification.query_type} 
        (Confidence: {classification.confidence:.1%})<br>
        <strong>Reasoning:</strong> {classification.reasoning}
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Time", f"{result['total_time']:.2f}s")
    with col2:
        vector_count = len(result['vector_results'].results) if result.get('vector_results') else 0
        st.metric("Vector Results", vector_count)
    with col3:
        graph_count = len(result['graph_results'].relationships) if result.get('graph_results') else 0
        st.metric("Graph Relations", graph_count)
    with col4:
        st.metric("Evidence Pieces", len(result['synthesis'].supporting_evidence))
    
    # Detailed information in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Sources", "🔗 Relationships", "🧠 Reasoning", "📊 Details"])
    
    with tab1:
        if show_sources and result['synthesis'].supporting_evidence:
            st.markdown("### Supporting Evidence")
            for i, evidence in enumerate(result['synthesis'].supporting_evidence):
                with st.expander(f"Evidence {i+1} - {evidence['type'].title()}"):
                    st.write(f"**Content:** {evidence['content']}")
                    if evidence['type'] == 'semantic':
                        st.write(f"**Relevance Score:** {evidence.get('score', 'N/A')}")
                        st.write(f"**Source:** {evidence.get('source', 'Unknown')}")
                    else:
                        st.write(f"**Confidence:** {evidence.get('confidence', 'N/A')}")
                        st.write(f"**Source:** {evidence.get('source', 'Knowledge Graph')}")
        else:
            st.info("No detailed source information available.")
    
    with tab2:
        if result.get('graph_results') and result['graph_results'].relationships:
            st.markdown("### Knowledge Graph Relationships")
            for i, rel in enumerate(result['graph_results'].relationships[:10]):  # Show top 10
                subject = rel.get('subject', {}).get('name', 'Unknown') if isinstance(rel.get('subject'), dict) else str(rel.get('subject', ''))
                predicate = rel.get('relationship', {}).get('type', 'RELATED_TO') if isinstance(rel.get('relationship'), dict) else 'RELATED_TO'
                obj = rel.get('object', {}).get('name', 'Unknown') if isinstance(rel.get('object'), dict) else str(rel.get('object', ''))
                
                st.write(f"{i+1}. **{subject}** → `{predicate}` → **{obj}**")
        else:
            st.info("No graph relationships found for this query.")
    
    with tab3:
        if show_reasoning:
            st.markdown("### Reasoning Process")
            st.write(f"**Classification Reasoning:** {classification.reasoning}")
            st.write(f"**Synthesis Reasoning:** {result['synthesis'].reasoning}")
            
            st.markdown("**Sources Used:**")
            sources_info = result['synthesis'].sources_used
            st.json(sources_info)
        else:
            st.info("Reasoning details hidden.")
    
    with tab4:
        st.markdown("### Technical Details")
        
        # Vector search details
        if result.get('vector_results'):
            with st.expander("Vector Search Details"):
                st.write(f"**Query:** {result['vector_results'].query}")
                st.write(f"**Results:** {len(result['vector_results'].results)}")
                st.write(f"**Retrieval Time:** {result['vector_results'].retrieval_time:.3f}s")
                
                if result['vector_results'].results:
                    st.write("**Top Results:**")
                    for i, res in enumerate(result['vector_results'].results[:3]):
                        st.write(f"{i+1}. Score: {res.score:.3f} - {res.text[:100]}...")
        
        # Graph search details
        if result.get('graph_results'):
            with st.expander("Graph Search Details"):
                st.write(f"**Query:** {result['graph_results'].query}")
                st.write(f"**Query Type:** {result['graph_results'].query_type}")
                st.write(f"**Entities:** {len(result['graph_results'].entities)}")
                st.write(f"**Relationships:** {len(result['graph_results'].relationships)}")
                st.write(f"**Retrieval Time:** {result['graph_results'].retrieval_time:.3f}s")

def conversation_history():
    """Display conversation history."""
    if st.session_state.conversation_history:
        st.markdown('<div class="section-header">💭 Conversation History</div>', unsafe_allow_html=True)
        
        for i, conv in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['query'][:50]}... - {conv['timestamp']}"):
                st.write(f"**Question:** {conv['query']}")
                st.write(f"**Answer:** {conv['result']['synthesis'].answer}")
                if st.button(f"Re-run Query", key=f"rerun_{i}"):
                    st.session_state.current_query = conv['query']
                    st.experimental_rerun()

def system_dashboard():
    """Display system status and statistics."""
    st.markdown('<div class="section-header">📊 System Dashboard</div>', unsafe_allow_html=True)
    
    # Get system statistics
    vector_store = st.session_state.vector_store_manager.get_store('main')
    vector_stats = vector_store.get_statistics()
    graph_stats = st.session_state.graph_store.get_graph_statistics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 Documents</h3>
            <h2>{}</h2>
            <p>Processed</p>
        </div>
        """.format(len(st.session_state.processed_documents)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Vector Store</h3>
            <h2>{}</h2>
            <p>Text Chunks</p>
        </div>
        """.format(vector_stats.get('total_documents', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🕸️ Graph Store</h3>
            <h2>{}</h2>
            <p>Total Nodes</p>
        </div>
        """.format(graph_stats.get('total_nodes', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔗 Relations</h3>
            <h2>{}</h2>
            <p>Total Edges</p>
        </div>
        """.format(graph_stats.get('total_relationships', 0)), unsafe_allow_html=True)
    
    # Detailed statistics
    with st.expander("📈 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Vector Database:**")
            st.json(vector_stats)
        
        with col2:
            st.markdown("**Graph Database:**")
            st.json(graph_stats)

def sidebar_content():
    """Render sidebar content."""
    st.sidebar.markdown("## 🛠️ System Controls")
    
    # System status
    if st.sidebar.button("🔄 Refresh System Status"):
        st.experimental_rerun()
    
    if st.sidebar.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")
    
    # Configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    with st.sidebar.expander("Vector Search Settings"):
        vector_top_k = st.slider("Max Vector Results", 1, 20, 10)
        vector_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3)
    
    with st.sidebar.expander("Graph Search Settings"):
        graph_max_hops = st.slider("Max Graph Hops", 1, 3, 2)
        graph_max_results = st.slider("Max Graph Results", 5, 50, 20)
    
    # Help and documentation
    st.sidebar.markdown("## 📚 Help & Documentation")
    
    with st.sidebar.expander("Query Examples"):
        st.markdown("""
        **Semantic Queries:**
        - What are the main financial risks?
        - Summarize the company's performance
        - Explain the market strategy
        
        **Relational Queries:**
        - Who is the CEO of Microsoft?
        - Which companies did Apple acquire?
        - List all partnerships
        
        **Hybrid Queries:**
        - Analyze performance and key relationships
        - What did executives say about acquisitions?
        """)
    
    with st.sidebar.expander("System Architecture"):
        st.markdown("""
        This system uses:
        - **Vector Search**: Semantic similarity using FAISS
        - **Graph Search**: Relationship traversal using Neo4j
        - **LLM Processing**: Gemma for entity/relation extraction
        - **Multi-Agent**: Intelligent query routing and synthesis
        """)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    sidebar_content()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Query Interface", "💭 History", "📊 Dashboard"])
    
    with tab1:
        document_upload_section()
    
    with tab2:
        query_interface()
    
    with tab3:
        conversation_history()
    
    with tab4:
        system_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🧠 Graph-Enhanced Agentic RAG for Corporate Intelligence<br>
        Built with Streamlit • Vector Search • Knowledge Graphs • Multi-Agent AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
