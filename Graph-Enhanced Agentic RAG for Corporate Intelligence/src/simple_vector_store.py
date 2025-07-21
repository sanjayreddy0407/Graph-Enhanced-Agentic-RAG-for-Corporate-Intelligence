"""Simplified vector store using sklearn for basic similarity search."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple vector store using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self.vectors = None
        self.documents = []
        self.metadata = []
        self.is_fitted = False
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store."""
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Add to storage
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Refit vectorizer with all documents
        self.vectors = self.vectorizer.fit_transform(self.documents)
        self.is_fitted = True
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if not self.is_fitted or len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            top_similarities = similarities[top_indices]
            
            results = []
            for i, similarity in zip(top_indices, top_similarities):
                if similarity > 0.01:  # Minimum similarity threshold
                    results.append({
                        'text': self.documents[i],
                        'metadata': self.metadata[i],
                        'similarity': float(similarity),
                        'index': int(i)
                    })
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.documents),
            'is_fitted': self.is_fitted,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_fitted else 0
        }

# For compatibility with existing code
VectorStore = SimpleVectorStore
