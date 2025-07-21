"""Embedding utilities and helper functions."""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class EmbeddingManager:
    """Manages different embedding models and provides utilities."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.dimension = 384  # Standard dimension for all-MiniLM-L6-v2
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get dimension from model
            sample_embedding = self.model.encode(["test"])
            self.dimension = sample_embedding.shape[1]
            
            logger.info(f"Embedding model loaded with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.dimension = 384  # Default fallback
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is None:
            # Return mock embeddings for development
            return self._mock_embeddings(texts)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            return self._mock_embeddings(texts)
    
    def _mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for development/testing."""
        np.random.seed(42)  # For reproducible results
        embeddings = []
        
        for text in texts:
            # Create deterministic but varied embeddings based on text
            seed = hash(text) % (2**31)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray,
                         top_k: int = 10) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Matrix of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: Matrix of embeddings to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        try:
            from sklearn.cluster import KMeans
            
            if len(embeddings) < n_clusters:
                logger.warning(f"Number of embeddings ({len(embeddings)}) is less than n_clusters ({n_clusters})")
                n_clusters = len(embeddings)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            return {
                'labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist(),
                'n_clusters': n_clusters,
                'inertia': kmeans.inertia_
            }
            
        except ImportError:
            logger.error("scikit-learn not available for clustering")
            return {
                'labels': [0] * len(embeddings),  # All in one cluster
                'centroids': [embeddings.mean(axis=0).tolist()],
                'n_clusters': 1,
                'error': 'scikit-learn not available'
            }
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {
                'labels': [0] * len(embeddings),
                'centroids': [embeddings.mean(axis=0).tolist()] if len(embeddings) > 0 else [],
                'n_clusters': 1,
                'error': str(e)
            }
    
    def reduce_dimensionality(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality of embeddings using PCA for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Target number of dimensions
            
        Returns:
            Reduced-dimension embeddings
        """
        try:
            from sklearn.decomposition import PCA
            
            if embeddings.shape[1] <= n_components:
                return embeddings
            
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {n_components} dimensions")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            return reduced_embeddings
            
        except ImportError:
            logger.error("scikit-learn not available for dimensionality reduction")
            # Return first n_components dimensions as fallback
            return embeddings[:, :n_components]
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return embeddings[:, :min(n_components, embeddings.shape[1])]

class TextPreprocessor:
    """Utilities for preprocessing text before embedding."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for better embeddings."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\'\/]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting (could be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text (simple frequency-based)."""
        import re
        from collections import Counter
        
        # Remove common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter and count
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(top_k)]

class EmbeddingCache:
    """Cache for storing and retrieving embeddings to avoid recomputation."""
    
    def __init__(self, cache_size: int = 10000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        text_hash = hash(text)
        if text_hash in self.cache:
            self.access_count[text_hash] = self.access_count.get(text_hash, 0) + 1
            return self.cache[text_hash]
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        text_hash = hash(text)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.cache_size:
            lru_hash = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_hash]
            del self.access_count[lru_hash]
        
        self.cache[text_hash] = embedding.copy()
        self.access_count[text_hash] = 1
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.cache_size,
            'usage_ratio': len(self.cache) / self.cache_size
        }

if __name__ == "__main__":
    # Test embedding utilities
    print("Testing Embedding Utilities...")
    
    # Test embedding manager
    embedding_manager = EmbeddingManager()
    
    # Test encoding
    texts = [
        "Microsoft reported strong revenue growth",
        "Apple launched new iPhone models",
        "Google expanded its cloud services"
    ]
    
    embeddings = embedding_manager.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    similarity = embedding_manager.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two texts: {similarity:.3f}")
    
    # Test text preprocessing
    preprocessor = TextPreprocessor()
    
    sample_text = "This is   a sample text with   multiple    spaces!!! And some??? punctuation."
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test sentence splitting
    long_text = "Microsoft is a technology company. It was founded by Bill Gates. The company is headquartered in Redmond, Washington."
    sentences = preprocessor.split_into_sentences(long_text)
    print(f"Sentences: {sentences}")
    
    # Test keyword extraction
    keywords = preprocessor.extract_keywords(long_text)
    print(f"Keywords: {keywords}")
    
    # Test caching
    cache = EmbeddingCache(cache_size=100)
    cache.put("test text", embeddings[0])
    cached_embedding = cache.get("test text")
    print(f"Cache test - Retrieved embedding shape: {cached_embedding.shape if cached_embedding is not None else 'None'}")
    
    print("Embedding utilities test completed successfully!")
