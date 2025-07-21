"""Configuration settings for the Graph-Enhanced Agentic RAG system."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class."""
    
    # Model Configuration
    LLAMA_MODEL_NAME = "llama3.2"  # Llama 3.2 via Ollama
    GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "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")
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast embedding model
    
    # Database Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # Vector Database Configuration
    VECTOR_DB_PATH = "data/vector_db"
    VECTOR_DIMENSION = 384  # For all-MiniLM-L6-v2
    
    # Document Processing Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_TOKENS_PER_CHUNK = 512
    
    # Agent Configuration
    MAX_RETRIEVAL_RESULTS = 10
    SIMILARITY_THRESHOLD = 0.7
    
    # UI Configuration
    PAGE_TITLE = "Graph-Enhanced Corporate Intelligence"
    PAGE_ICON = "🧠"
    
    # File Paths
    DATA_DIR = "data"
    DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    @classmethod
    def get_neo4j_config(cls) -> Dict[str, Any]:
        """Get Neo4j configuration dictionary."""
        return {
            "uri": cls.NEO4J_URI,
            "user": cls.NEO4J_USER,
            "password": cls.NEO4J_PASSWORD
        }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.DOCUMENTS_DIR,
            cls.PROCESSED_DIR,
            cls.VECTOR_DB_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Entity and Relation Extraction Prompts (Optimized for Llama 3.2)
ENTITY_EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at extracting entities from corporate documents. Extract entities and return them in valid JSON format only.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following types of entities from the given text:

1. Companies/Organizations
2. People (CEOs, executives, etc.)
3. Financial metrics (revenue, profit, etc.)
4. Products/Services
5. Dates/Time periods
6. Locations

Return ONLY valid JSON in this exact format:
{
  "companies": ["company1", "company2"],
  "people": ["person1", "person2"],
  "financials": ["metric1", "metric2"],
  "products": ["product1", "product2"],
  "dates": ["date1", "date2"],
  "locations": ["location1", "location2"]
}

Text to analyze:
{text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

RELATION_EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at extracting relationships between entities in corporate documents. Extract relationships and return them in valid JSON format only.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract relationships between entities in the format (Subject, Predicate, Object).

Common relationship types:
- CEO_OF, EXECUTIVE_OF
- ACQUIRED, MERGED_WITH
- PARTNER_WITH, SUPPLIER_TO
- REPORTED, ANNOUNCED
- LOCATED_IN, BASED_IN

Return ONLY valid JSON in this exact format:
{
  "relationships": [
    {"subject": "Entity1", "predicate": "RELATION_TYPE", "object": "Entity2"},
    {"subject": "Entity3", "predicate": "RELATION_TYPE", "object": "Entity4"}
  ]
}

Text to analyze:
{text}

Previously extracted entities:
{entities}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

QUERY_CLASSIFICATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query classifier for a corporate intelligence system.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify the user query into one of these categories:

1. SEMANTIC - Questions about general topics, summaries, explanations
2. RELATIONAL - Questions about specific connections, acquisitions, hierarchies
3. HYBRID - Questions that need both semantic understanding and relational data

Examples:
- "What challenges did Microsoft face?" → SEMANTIC
- "Which companies did Apple acquire?" → RELATIONAL  
- "Summarize Amazon's performance and key partnerships" → HYBRID

Return only the category name: SEMANTIC, RELATIONAL, or HYBRID

Query: {query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# Initialize configuration
config = Config()
config.ensure_directories()
