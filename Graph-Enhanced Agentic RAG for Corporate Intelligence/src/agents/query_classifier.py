"""Query classification agent for routing queries to appropriate backends."""

import re
from typing import Literal, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

QueryType = Literal["SEMANTIC", "RELATIONAL", "HYBRID"]

@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_approaches: Dict[str, bool]

class QueryClassifierAgent:
    """Agent that classifies user queries into semantic, relational, or hybrid categories."""
    
    def __init__(self):
        # Keywords that indicate relational queries
        self.relational_keywords = [
            'acquired', 'acquisition', 'bought', 'purchased', 'merged', 'merger',
            'subsidiary', 'subsidiaries', 'owns', 'owned by',
            'ceo', 'executive', 'president', 'chairman', 'founder',
            'partner', 'partnership', 'collaboration', 'joint venture',
            'supplier', 'customer', 'client',
            'relationship', 'connection', 'linked to', 'associated with',
            'who is', 'who are', 'which companies', 'list all',
            'affiliated', 'board member', 'director'
        ]
        
        # Keywords that indicate semantic queries
        self.semantic_keywords = [
            'summarize', 'summary', 'overview', 'explain', 'describe',
            'challenges', 'risks', 'opportunities', 'performance',
            'strategy', 'outlook', 'trends', 'analysis',
            'what', 'why', 'how', 'when',
            'revenue', 'profit', 'earnings', 'financial performance',
            'market', 'industry', 'competition', 'competitive',
            'growth', 'decline', 'increase', 'decrease'
        ]
        
        # Keywords that suggest hybrid queries
        self.hybrid_keywords = [
            'performance and partnerships', 'strategy and acquisitions',
            'growth and relationships', 'financial performance and key',
            'overview and connections', 'analysis and partnerships'
        ]
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a user query into SEMANTIC, RELATIONAL, or HYBRID.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification object with type, confidence, and reasoning
        """
        query_lower = query.lower()
        
        # Check for hybrid patterns first (more specific)
        hybrid_score = self._calculate_keyword_score(query_lower, self.hybrid_keywords)
        relational_score = self._calculate_keyword_score(query_lower, self.relational_keywords)
        semantic_score = self._calculate_keyword_score(query_lower, self.semantic_keywords)
        
        # Additional pattern matching
        relational_score += self._check_relational_patterns(query_lower)
        semantic_score += self._check_semantic_patterns(query_lower)
        
        # Determine classification
        scores = {
            'RELATIONAL': relational_score,
            'SEMANTIC': semantic_score,
            'HYBRID': hybrid_score
        }
        
        # If hybrid score is significant or both semantic and relational scores are high
        if hybrid_score > 0.3 or (relational_score > 0.4 and semantic_score > 0.4):
            query_type = "HYBRID"
            confidence = min(0.95, hybrid_score + 0.2)
            reasoning = "Query contains both semantic and relational elements"
            approaches = {"use_vector": True, "use_graph": True}
        elif relational_score > semantic_score and relational_score > 0.3:
            query_type = "RELATIONAL"
            confidence = min(0.95, relational_score)
            reasoning = f"Query focuses on entity relationships (score: {relational_score:.2f})"
            approaches = {"use_vector": False, "use_graph": True}
        else:
            query_type = "SEMANTIC"
            confidence = min(0.95, max(0.6, semantic_score))  # Default to semantic with reasonable confidence
            reasoning = f"Query requires semantic understanding (score: {semantic_score:.2f})"
            approaches = {"use_vector": True, "use_graph": False}
        
        logger.info(f"Classified query as {query_type} with confidence {confidence:.2f}")
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_approaches=approaches
        )
    
    def _calculate_keyword_score(self, query: str, keywords: list) -> float:
        """Calculate score based on keyword matches."""
        matches = 0
        total_weight = 0
        
        for keyword in keywords:
            if keyword in query:
                matches += 1
                # Give higher weight to longer, more specific keywords
                weight = len(keyword.split())
                total_weight += weight
        
        if not keywords:
            return 0.0
        
        # Normalize score
        base_score = matches / len(keywords)
        weighted_score = total_weight / (len(keywords) * 2)  # Assuming average 2 words per keyword
        
        return min(1.0, (base_score + weighted_score) / 2)
    
    def _check_relational_patterns(self, query: str) -> float:
        """Check for patterns that suggest relational queries."""
        patterns = [
            r'\b(who|which)\s+(companies?|people|executives?)\b',
            r'\b(list|show)\s+(all|the)\b',
            r'\b(connections?|relationships?)\s+between\b',
            r'\b(acquired|owned|merged)\s+(by|with)\b',
            r'\bCEO\s+of\b',
            r'\bsubsidiaries?\s+of\b',
            r'\bpartnership\s+with\b'
        ]
        
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.2
        
        return min(1.0, score)
    
    def _check_semantic_patterns(self, query: str) -> float:
        """Check for patterns that suggest semantic queries."""
        patterns = [
            r'\b(what|why|how)\s+(are|is|did|does)\b',
            r'\bsummariz[e|ing]\b',
            r'\b(explain|describe|tell me about)\b',
            r'\b(performance|financial|revenue)\b',
            r'\b(challenges?|risks?|opportunities?)\b',
            r'\b(trends?|analysis|outlook)\b'
        ]
        
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.2
        
        return min(1.0, score)
    
    def get_query_insights(self, query: str) -> Dict[str, Any]:
        """Get detailed insights about the query for debugging/explanation."""
        query_lower = query.lower()
        
        matched_relational = [kw for kw in self.relational_keywords if kw in query_lower]
        matched_semantic = [kw for kw in self.semantic_keywords if kw in query_lower]
        matched_hybrid = [kw for kw in self.hybrid_keywords if kw in query_lower]
        
        return {
            'original_query': query,
            'matched_keywords': {
                'relational': matched_relational,
                'semantic': matched_semantic,
                'hybrid': matched_hybrid
            },
            'query_length': len(query.split()),
            'contains_question_words': any(word in query_lower for word in ['what', 'why', 'how', 'when', 'where', 'who']),
            'contains_list_words': any(word in query_lower for word in ['list', 'show', 'enumerate', 'all'])
        }

# Example usage and predefined query patterns
class PredefinedQueries:
    """Collection of predefined query examples for testing and demonstration."""
    
    SEMANTIC_QUERIES = [
        "What were the main challenges Microsoft faced in 2023?",
        "Summarize Amazon's financial performance last quarter",
        "Describe Apple's market strategy in the smartphone segment",
        "Why did Tesla's stock price decline?",
        "How has Google's cloud business grown?",
        "What risks does Meta mention in their annual report?",
        "Explain Netflix's content strategy",
        "What are the key trends in the tech industry?"
    ]
    
    RELATIONAL_QUERIES = [
        "Which companies did Microsoft acquire in 2023?",
        "Who is the CEO of Apple?",
        "List all subsidiaries of Amazon",
        "What partnerships does Google have?",
        "Which executives work at Meta?",
        "Show all companies that Tesla has invested in",
        "Who are Netflix's key suppliers?",
        "List all board members of Oracle"
    ]
    
    HYBRID_QUERIES = [
        "Summarize Apple's financial performance and key partnerships",
        "What did Microsoft's CEO say about AI and who are their main competitors?",
        "Describe Amazon's growth strategy and recent acquisitions",
        "Explain Google's cloud performance and partnership strategy",
        "What are Meta's key challenges and who leads their AI division?",
        "Analyze Tesla's market position and supplier relationships",
        "Overview of Netflix's content strategy and production partnerships",
        "Discuss Oracle's financial health and recent merger activities"
    ]

if __name__ == "__main__":
    # Test the classifier
    classifier = QueryClassifierAgent()
    
    # Test different types of queries
    test_queries = [
        "What challenges did Microsoft face in 2023?",  # SEMANTIC
        "Which companies did Apple acquire?",  # RELATIONAL
        "Summarize Amazon's performance and key partnerships"  # HYBRID
    ]
    
    print("Query Classification Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        result = classifier.classify_query(query)
        insights = classifier.get_query_insights(query)
        
        print(f"\nQuery: {query}")
        print(f"Type: {result.query_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Approaches: {result.suggested_approaches}")
        print(f"Matched keywords: {insights['matched_keywords']}")
        print("-" * 30)
    
    print("\nClassifier test completed successfully!")
