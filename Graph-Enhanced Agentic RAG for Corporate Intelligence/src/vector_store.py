"""Vector database operations using FAISS for semantic search."""

import os
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from vector database."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str

class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _create_index(self, vectors: np.ndarray):
        """Create FAISS index from vectors."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match expected {self.dimension}")
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents to add
            metadata: Optional metadata for each document
        """
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
        
        if len(documents) != len(metadata):
            raise ValueError("Documents and metadata must have the same length")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype(np.float32)
        
        # Create or update index
        if self.index is None:
            self._create_index(embeddings)
        else:
            # Normalize new vectors
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any] = None):
        """
        Add processed document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with 'text' field
            document_metadata: Metadata about the source document
        """
        documents = []
        metadata = []
        
        if document_metadata is None:
            document_metadata = {}
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            
            chunk_metadata = {
                'chunk_id': f"{document_metadata.get('filename', 'unknown')}_{i}",
                'chunk_index': i,
                'word_count': chunk.get('word_count', 0),
                'start_idx': chunk.get('start_idx', 0),
                'end_idx': chunk.get('end_idx', 0),
                **document_metadata
            }
            metadata.append(chunk_metadata)
        
        self.add_documents(documents, metadata)
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:  # Filter out invalid indices and low scores
                results.append(SearchResult(
                    text=self.documents[idx],
                    score=float(score),
                    metadata=self.metadata[idx],
                    chunk_id=self.metadata[idx].get('chunk_id', str(idx))
                ))
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def search_with_filter(self, query: str, filter_func: callable, top_k: int = 10) -> List[SearchResult]:
        """
        Search with custom filtering function.
        
        Args:
            query: Search query
            filter_func: Function to filter metadata (takes metadata dict, returns bool)
            top_k: Number of results to return
        """
        # Get more results initially to account for filtering
        initial_results = self.search(query, top_k * 3)
        
        # Apply filter
        filtered_results = [result for result in initial_results if filter_func(result.metadata)]
        
        # Return top_k filtered results
        return filtered_results[:top_k]
    
    def save_index(self, index_path: str):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # Save documents and metadata
        with open(f"{index_path}_data.pkl", "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'embedding_model_name': self.embedding_model_name
            }, f)
        
        logger.info(f"Vector store saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load documents and metadata
            with open(f"{index_path}_data.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                
                # Verify model compatibility
                if data.get('embedding_model_name') != self.embedding_model_name:
                    logger.warning(f"Loaded index was created with {data.get('embedding_model_name')}, "
                                 f"but current model is {self.embedding_model_name}")
            
            logger.info(f"Vector store loaded from {index_path} with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'embedding_model': self.embedding_model_name,
            'memory_usage_mb': self.index.sa_code_size() / (1024*1024) if self.index else 0
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        self.index = None
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")

class VectorStoreManager:
    """Manager class for handling multiple vector stores."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.stores = {}
    
    def get_store(self, store_name: str, embedding_model: str = "all-MiniLM-L6-v2") -> VectorStore:
        """Get or create a vector store."""
        if store_name not in self.stores:
            self.stores[store_name] = VectorStore(embedding_model)
            
            # Try to load existing index
            index_path = os.path.join(self.base_path, store_name)
            if os.path.exists(f"{index_path}.faiss"):
                try:
                    self.stores[store_name].load_index(index_path)
                except Exception as e:
                    logger.warning(f"Failed to load existing index for {store_name}: {e}")
        
        return self.stores[store_name]
    
    def save_all_stores(self):
        """Save all vector stores."""
        for store_name, store in self.stores.items():
            index_path = os.path.join(self.base_path, store_name)
            store.save_index(index_path)
    
    def list_stores(self) -> List[str]:
        """List all available stores."""
        return list(self.stores.keys())

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()
    
    # Sample documents
    documents = [
        "Microsoft reported strong quarterly earnings with Azure growth.",
        "Apple launched new iPhone models with advanced AI capabilities.",
        "Amazon's cloud services continue to dominate the market.",
        "Google announced new AI initiatives and partnerships."
    ]
    
    metadata = [
        {"company": "Microsoft", "type": "earnings"},
        {"company": "Apple", "type": "product"},
        {"company": "Amazon", "type": "market"},
        {"company": "Google", "type": "AI"}
    ]
    
    # Add documents
    vector_store.add_documents(documents, metadata)
    
    # Search
    results = vector_store.search("AI technology developments", top_k=3)
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"Score: {result.score:.3f} - {result.text[:100]}...")
    
    # Save and load example
    # vector_store.save_index("data/vector_db/test_index")
    
    print("Vector store example completed successfully!")
