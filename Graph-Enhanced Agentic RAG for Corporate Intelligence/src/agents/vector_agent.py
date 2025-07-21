"""Vector retrieval agent for semantic search operations."""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from src.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class VectorRetrievalResult:
    """Result from vector-based retrieval."""
    results: List[SearchResult]
    query: str
    retrieval_time: float
    metadata: Dict[str, Any]

class VectorAgent:
    """Agent responsible for vector-based semantic retrieval."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.default_top_k = 10
        self.default_threshold = 0.3
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None, 
                threshold: Optional[float] = None,
                filters: Optional[Dict[str, Any]] = None) -> VectorRetrievalResult:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional filters for metadata
            
        Returns:
            VectorRetrievalResult with search results and metadata
        """
        import time
        start_time = time.time()
        
        # Use defaults if not provided
        top_k = top_k or self.default_top_k
        threshold = threshold or self.default_threshold
        
        logger.info(f"Vector retrieval for query: '{query[:50]}...' (top_k={top_k})")
        
        try:
            if filters:
                # Use filtered search if filters are provided
                def filter_func(metadata: Dict[str, Any]) -> bool:
                    return all(
                        metadata.get(key) == value 
                        for key, value in filters.items()
                    )
                
                results = self.vector_store.search_with_filter(
                    query=query,
                    filter_func=filter_func,
                    top_k=top_k
                )
            else:
                # Standard semantic search
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    threshold=threshold
                )
            
            retrieval_time = time.time() - start_time
            
            logger.info(f"Vector retrieval completed: {len(results)} results in {retrieval_time:.3f}s")
            
            return VectorRetrievalResult(
                results=results,
                query=query,
                retrieval_time=retrieval_time,
                metadata={
                    'top_k': top_k,
                    'threshold': threshold,
                    'filters_applied': filters is not None,
                    'total_results': len(results)
                }
            )
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return VectorRetrievalResult(
                results=[],
                query=query,
                retrieval_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def retrieve_with_context(self, 
                             query: str, 
                             context_queries: List[str],
                             top_k: Optional[int] = None) -> VectorRetrievalResult:
        """
        Retrieve with additional context queries for better results.
        
        Args:
            query: Main search query
            context_queries: Additional related queries for context
            top_k: Number of results to return
            
        Returns:
            VectorRetrievalResult with enhanced search results
        """
        logger.info(f"Context-enhanced retrieval for '{query[:50]}...' with {len(context_queries)} context queries")
        
        # Combine main query with context
        all_queries = [query] + context_queries
        combined_query = " ".join(all_queries)
        
        return self.retrieve(combined_query, top_k=top_k)
    
    def retrieve_by_document(self, 
                           query: str, 
                           document_name: str, 
                           top_k: Optional[int] = None) -> VectorRetrievalResult:
        """
        Retrieve results from a specific document.
        
        Args:
            query: Search query
            document_name: Name of the document to search within
            top_k: Number of results to return
            
        Returns:
            VectorRetrievalResult filtered to specific document
        """
        filters = {'filename': document_name}
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def retrieve_by_section(self, 
                          query: str, 
                          section_type: str, 
                          top_k: Optional[int] = None) -> VectorRetrievalResult:
        """
        Retrieve results from specific document sections.
        
        Args:
            query: Search query
            section_type: Type of section (e.g., 'risk_factors', 'financials')
            top_k: Number of results to return
            
        Returns:
            VectorRetrievalResult filtered to specific section type
        """
        filters = {'section_type': section_type}
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def get_similar_chunks(self, 
                          reference_text: str, 
                          top_k: Optional[int] = None,
                          exclude_exact_matches: bool = True) -> VectorRetrievalResult:
        """
        Find chunks similar to a reference text.
        
        Args:
            reference_text: Text to find similar content for
            top_k: Number of results to return
            exclude_exact_matches: Whether to exclude exact text matches
            
        Returns:
            VectorRetrievalResult with similar chunks
        """
        result = self.retrieve(reference_text, top_k=top_k)
        
        if exclude_exact_matches:
            # Filter out results that are too similar (likely exact matches)
            filtered_results = [
                r for r in result.results 
                if r.text.strip() != reference_text.strip()
            ]
            result.results = filtered_results
            result.metadata['exact_matches_excluded'] = True
        
        return result
    
    def explain_retrieval(self, result: VectorRetrievalResult) -> Dict[str, Any]:
        """
        Provide explanation for retrieval results.
        
        Args:
            result: VectorRetrievalResult to explain
            
        Returns:
            Dictionary with explanation details
        """
        if not result.results:
            return {
                'explanation': 'No results found for the query',
                'suggestions': [
                    'Try a broader query',
                    'Check spelling and terminology',
                    'Ensure documents are loaded in the vector store'
                ]
            }
        
        # Analyze result quality
        scores = [r.score for r in result.results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Analyze source diversity
        sources = set()
        for r in result.results:
            if 'filename' in r.metadata:
                sources.add(r.metadata['filename'])
        
        explanation = {
            'total_results': len(result.results),
            'score_statistics': {
                'average': round(avg_score, 3),
                'minimum': round(min_score, 3),
                'maximum': round(max_score, 3)
            },
            'source_diversity': {
                'unique_sources': len(sources),
                'source_names': list(sources)
            },
            'retrieval_time': round(result.retrieval_time, 3),
            'quality_assessment': self._assess_result_quality(result)
        }
        
        return explanation
    
    def _assess_result_quality(self, result: VectorRetrievalResult) -> str:
        """Assess the quality of retrieval results."""
        if not result.results:
            return "No results"
        
        scores = [r.score for r in result.results]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.8:
            return "High quality - very relevant results"
        elif avg_score > 0.6:
            return "Good quality - mostly relevant results"
        elif avg_score > 0.4:
            return "Moderate quality - some relevant results"
        else:
            return "Low quality - results may not be very relevant"
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store and retrieval capabilities."""
        if self.vector_store:
            return self.vector_store.get_statistics()
        return {}

class SemanticQueryProcessor:
    """Helper class for processing and enhancing semantic queries."""
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Expand a query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variations
        """
        # Simple expansion based on business terminology
        expansions = [query]  # Start with original
        
        # Financial terms expansion
        financial_synonyms = {
            'revenue': ['sales', 'income', 'earnings', 'turnover'],
            'profit': ['earnings', 'income', 'net income', 'surplus'],
            'growth': ['increase', 'expansion', 'rise', 'improvement'],
            'decline': ['decrease', 'reduction', 'fall', 'drop'],
            'performance': ['results', 'achievements', 'outcomes', 'metrics']
        }
        
        query_lower = query.lower()
        for term, synonyms in financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    expansions.append(expanded)
        
        return expansions[:5]  # Limit to avoid too many variations
    
    @staticmethod
    def extract_key_concepts(query: str) -> List[str]:
        """
        Extract key concepts from a query for focused retrieval.
        
        Args:
            query: Query to analyze
            
        Returns:
            List of key concepts
        """
        import re
        
        # Remove common words
        stop_words = {
            'what', 'why', 'how', 'when', 'where', 'who', 'which', 'that',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
        }
        
        # Extract words, keeping important business terms
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize business-relevant terms
        business_terms = {
            'revenue', 'profit', 'earnings', 'acquisition', 'merger', 'ceo',
            'company', 'corporation', 'growth', 'performance', 'strategy',
            'market', 'industry', 'financial', 'business', 'operations'
        }
        
        prioritized = []
        regular = []
        
        for concept in key_concepts:
            if concept in business_terms:
                prioritized.append(concept)
            else:
                regular.append(concept)
        
        return prioritized + regular[:5]  # Business terms first, then others

if __name__ == "__main__":
    # Mock testing without actual vector store
    class MockVectorStore:
        def search(self, query, top_k=10, threshold=0.3):
            # Mock search results
            return [
                SearchResult(
                    text=f"Mock result for query: {query}",
                    score=0.85,
                    metadata={'filename': 'test_doc.pdf', 'chunk_id': 'chunk_1'},
                    chunk_id='chunk_1'
                )
            ]
        
        def search_with_filter(self, query, filter_func, top_k=10):
            return self.search(query, top_k)
        
        def get_statistics(self):
            return {'total_documents': 100, 'dimension': 384}
    
    # Test vector agent
    mock_store = MockVectorStore()
    agent = VectorAgent(mock_store)
    
    test_query = "What are the financial risks mentioned in the annual report?"
    result = agent.retrieve(test_query)
    
    print(f"Vector Agent Test:")
    print(f"Query: {test_query}")
    print(f"Results: {len(result.results)}")
    print(f"Time: {result.retrieval_time:.3f}s")
    
    explanation = agent.explain_retrieval(result)
    print(f"Quality: {explanation['quality_assessment']}")
    
    # Test query processor
    processor = SemanticQueryProcessor()
    expansions = processor.expand_query("Company revenue growth")
    key_concepts = processor.extract_key_concepts("What are the main financial risks for Apple?")
    
    print(f"\nQuery expansions: {expansions}")
    print(f"Key concepts: {key_concepts}")
    
    print("Vector agent test completed successfully!")
