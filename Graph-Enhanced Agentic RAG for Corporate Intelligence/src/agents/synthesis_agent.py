"""Synthesis agent for combining and generating comprehensive answers."""

from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from src.agents.vector_agent import VectorRetrievalResult
from src.agents.graph_agent import GraphRetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    """Result from synthesis agent combining multiple retrieval sources."""
    answer: str
    confidence: float
    sources_used: Dict[str, Any]
    reasoning: str
    supporting_evidence: List[Dict[str, Any]]
    synthesis_time: float

class SynthesisAgent:
    """Agent responsible for synthesizing answers from multiple retrieval sources."""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model  # Could be Gemma or other LLM
        self.max_context_length = 4000  # Maximum context for answer generation
        self.confidence_threshold = 0.6
    
    def synthesize_answer(self,
                         query: str,
                         vector_results: Optional[VectorRetrievalResult] = None,
                         graph_results: Optional[GraphRetrievalResult] = None,
                         additional_context: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize a comprehensive answer from multiple sources.
        
        Args:
            query: Original user query
            vector_results: Results from vector/semantic search
            graph_results: Results from graph/relational search
            additional_context: Any additional context to consider
            
        Returns:
            SynthesisResult with synthesized answer
        """
        import time
        start_time = time.time()
        
        logger.info(f"Synthesizing answer for query: '{query[:50]}...'")
        
        try:
            # Prepare context from all sources
            context = self._prepare_context(vector_results, graph_results, additional_context)
            
            # Determine synthesis strategy based on available sources
            strategy = self._determine_strategy(vector_results, graph_results)
            
            # Generate answer based on strategy
            if strategy == "vector_only":
                answer = self._synthesize_vector_only(query, vector_results, context)
            elif strategy == "graph_only":
                answer = self._synthesize_graph_only(query, graph_results, context)
            elif strategy == "hybrid":
                answer = self._synthesize_hybrid(query, vector_results, graph_results, context)
            else:
                answer = self._synthesize_fallback(query, context)
            
            # Calculate confidence based on source quality and answer coherence
            confidence = self._calculate_confidence(answer, vector_results, graph_results)
            
            # Extract supporting evidence
            evidence = self._extract_evidence(vector_results, graph_results)
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(strategy, vector_results, graph_results)
            
            synthesis_time = time.time() - start_time
            
            logger.info(f"Answer synthesized in {synthesis_time:.3f}s with confidence {confidence:.2f}")
            
            return SynthesisResult(
                answer=answer,
                confidence=confidence,
                sources_used=self._get_sources_summary(vector_results, graph_results),
                reasoning=reasoning,
                supporting_evidence=evidence,
                synthesis_time=synthesis_time
            )
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return SynthesisResult(
                answer="I apologize, but I encountered an error while processing your query. Please try rephrasing your question or check if the relevant documents are available.",
                confidence=0.0,
                sources_used={'error': str(e)},
                reasoning="Error in synthesis process",
                supporting_evidence=[],
                synthesis_time=time.time() - start_time
            )
    
    def _prepare_context(self,
                        vector_results: Optional[VectorRetrievalResult],
                        graph_results: Optional[GraphRetrievalResult],
                        additional_context: Optional[str]) -> str:
        """Prepare context string from all available sources."""
        context_parts = []
        
        # Add vector search context
        if vector_results and vector_results.results:
            context_parts.append("=== SEMANTIC SEARCH RESULTS ===")
            for i, result in enumerate(vector_results.results[:5]):  # Top 5 results
                context_parts.append(f"Result {i+1} (Score: {result.score:.3f}):")
                context_parts.append(result.text[:500] + "..." if len(result.text) > 500 else result.text)
                context_parts.append("")
        
        # Add graph search context
        if graph_results and (graph_results.entities or graph_results.relationships):
            context_parts.append("=== GRAPH RELATIONSHIPS ===")
            
            # Add key relationships
            for i, rel in enumerate(graph_results.relationships[:10]):  # Top 10 relationships
                if isinstance(rel, dict):
                    subject = self._extract_entity_name(rel.get('subject', {}))
                    predicate = self._extract_relation_type(rel.get('relationship', {}))
                    obj = self._extract_entity_name(rel.get('object', {}))
                    context_parts.append(f"Relationship {i+1}: {subject} {predicate} {obj}")
            
            context_parts.append("")
        
        # Add additional context if provided
        if additional_context:
            context_parts.append("=== ADDITIONAL CONTEXT ===")
            context_parts.append(additional_context)
            context_parts.append("")
        
        # Combine and limit context length
        full_context = "\n".join(context_parts)
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "... [truncated]"
        
        return full_context
    
    def _determine_strategy(self,
                           vector_results: Optional[VectorRetrievalResult],
                           graph_results: Optional[GraphRetrievalResult]) -> str:
        """Determine synthesis strategy based on available results."""
        has_vector = vector_results and vector_results.results
        has_graph = graph_results and (graph_results.entities or graph_results.relationships)
        
        if has_vector and has_graph:
            return "hybrid"
        elif has_vector:
            return "vector_only"
        elif has_graph:
            return "graph_only"
        else:
            return "fallback"
    
    def _synthesize_vector_only(self,
                               query: str,
                               vector_results: VectorRetrievalResult,
                               context: str) -> str:
        """Synthesize answer using only vector search results."""
        if not vector_results.results:
            return "I couldn't find relevant information in the documents to answer your query."
        
        # Use LLM if available, otherwise use template-based approach
        if self.llm_model:
            return self._llm_synthesize(query, context, "semantic")
        else:
            return self._template_synthesize_vector(query, vector_results)
    
    def _synthesize_graph_only(self,
                              query: str,
                              graph_results: GraphRetrievalResult,
                              context: str) -> str:
        """Synthesize answer using only graph search results."""
        if not graph_results.relationships and not graph_results.entities:
            return "I couldn't find relevant relationships in the knowledge graph to answer your query."
        
        # Use LLM if available, otherwise use template-based approach
        if self.llm_model:
            return self._llm_synthesize(query, context, "relational")
        else:
            return self._template_synthesize_graph(query, graph_results)
    
    def _synthesize_hybrid(self,
                          query: str,
                          vector_results: VectorRetrievalResult,
                          graph_results: GraphRetrievalResult,
                          context: str) -> str:
        """Synthesize answer combining both vector and graph results."""
        # Use LLM if available, otherwise use template-based approach
        if self.llm_model:
            return self._llm_synthesize(query, context, "hybrid")
        else:
            return self._template_synthesize_hybrid(query, vector_results, graph_results)
    
    def _synthesize_fallback(self, query: str, context: str) -> str:
        """Fallback synthesis when no results are available."""
        return (
            f"I don't have sufficient information to answer your query about '{query}'. "
            "This could be because:\n"
            "• The relevant documents haven't been uploaded yet\n"
            "• The information isn't present in the available documents\n"
            "• The query might need to be rephrased for better results"
        )
    
    def _llm_synthesize(self, query: str, context: str, synthesis_type: str) -> str:
        """Use LLM to synthesize answer (placeholder for actual LLM integration)."""
        # This would integrate with Gemma or another LLM
        # For now, return a placeholder
        prompt = self._create_synthesis_prompt(query, context, synthesis_type)
        
        # Mock LLM response - replace with actual LLM call
        return (
            f"Based on the available {synthesis_type} information, here's what I found regarding your query about {query}:\n\n"
            f"[This would be the actual LLM-generated response using the context: {context[:100]}...]"
        )
    
    def _create_synthesis_prompt(self, query: str, context: str, synthesis_type: str) -> str:
        """Create appropriate prompt for LLM synthesis."""
        base_prompt = f"""
You are a corporate intelligence assistant. Answer the user's query using the provided context.

Query: {query}

Available Context:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the context
- Cite specific information when possible
- If information is incomplete, acknowledge limitations
- Focus on {synthesis_type} aspects of the query
- Be professional and concise

Answer:"""
        
        return base_prompt
    
    def _template_synthesize_vector(self, query: str, vector_results: VectorRetrievalResult) -> str:
        """Template-based synthesis for vector-only results."""
        if not vector_results.results:
            return "No relevant information found."
        
        answer_parts = []
        answer_parts.append(f"Based on the available documents, here's what I found regarding '{query}':")
        answer_parts.append("")
        
        # Summarize key findings from top results
        for i, result in enumerate(vector_results.results[:3]):  # Top 3 results
            summary = result.text[:200] + "..." if len(result.text) > 200 else result.text
            answer_parts.append(f"Finding {i+1}: {summary}")
            if result.metadata.get('filename'):
                answer_parts.append(f"Source: {result.metadata['filename']}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _template_synthesize_graph(self, query: str, graph_results: GraphRetrievalResult) -> str:
        """Template-based synthesis for graph-only results."""
        answer_parts = []
        
        if graph_results.relationships:
            answer_parts.append(f"Based on the relationship data, here are the relevant connections for '{query}':")
            answer_parts.append("")
            
            # Group relationships by type
            rel_groups = {}
            for rel in graph_results.relationships[:10]:  # Top 10
                rel_type = self._extract_relation_type(rel.get('relationship', {}))
                if rel_type not in rel_groups:
                    rel_groups[rel_type] = []
                rel_groups[rel_type].append(rel)
            
            for rel_type, rels in rel_groups.items():
                answer_parts.append(f"{rel_type} relationships:")
                for rel in rels[:5]:  # Top 5 per type
                    subject = self._extract_entity_name(rel.get('subject', {}))
                    obj = self._extract_entity_name(rel.get('object', {}))
                    answer_parts.append(f"• {subject} → {obj}")
                answer_parts.append("")
        
        if not answer_parts:
            answer_parts.append("No relevant relationships found for your query.")
        
        return "\n".join(answer_parts)
    
    def _template_synthesize_hybrid(self,
                                   query: str,
                                   vector_results: VectorRetrievalResult,
                                   graph_results: GraphRetrievalResult) -> str:
        """Template-based synthesis combining both sources."""
        answer_parts = []
        answer_parts.append(f"Here's a comprehensive answer to '{query}' based on both document content and relationship data:")
        answer_parts.append("")
        
        # Add semantic findings
        if vector_results and vector_results.results:
            answer_parts.append("Key Information from Documents:")
            for i, result in enumerate(vector_results.results[:2]):  # Top 2
                summary = result.text[:150] + "..." if len(result.text) > 150 else result.text
                answer_parts.append(f"• {summary}")
            answer_parts.append("")
        
        # Add relational findings
        if graph_results and graph_results.relationships:
            answer_parts.append("Relevant Relationships:")
            for i, rel in enumerate(graph_results.relationships[:5]):  # Top 5
                subject = self._extract_entity_name(rel.get('subject', {}))
                predicate = self._extract_relation_type(rel.get('relationship', {}))
                obj = self._extract_entity_name(rel.get('object', {}))
                answer_parts.append(f"• {subject} {predicate} {obj}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _extract_entity_name(self, entity: Dict[str, Any]) -> str:
        """Extract entity name from graph result."""
        if isinstance(entity, dict):
            return entity.get('name', str(entity))
        return str(entity)
    
    def _extract_relation_type(self, relationship: Dict[str, Any]) -> str:
        """Extract relationship type from graph result."""
        if isinstance(relationship, dict):
            return relationship.get('type', 'RELATED_TO')
        return 'RELATED_TO'
    
    def _calculate_confidence(self,
                             answer: str,
                             vector_results: Optional[VectorRetrievalResult],
                             graph_results: Optional[GraphRetrievalResult]) -> float:
        """Calculate confidence score for the synthesized answer."""
        confidence = 0.5  # Base confidence
        
        # Factor in vector results quality
        if vector_results and vector_results.results:
            scores = [r.score for r in vector_results.results[:5]]  # Top 5
            avg_vector_score = sum(scores) / len(scores)
            confidence += avg_vector_score * 0.3
        
        # Factor in graph results quality
        if graph_results and graph_results.relationships:
            # Simple heuristic based on number of relationships found
            rel_count = len(graph_results.relationships)
            graph_score = min(1.0, rel_count / 10.0)  # Normalize to max 1.0
            confidence += graph_score * 0.2
        
        # Penalize very short or generic answers
        if len(answer.split()) < 10:
            confidence *= 0.7
        
        # Penalize error messages
        if "error" in answer.lower() or "couldn't find" in answer.lower():
            confidence *= 0.3
        
        return min(1.0, confidence)
    
    def _extract_evidence(self,
                         vector_results: Optional[VectorRetrievalResult],
                         graph_results: Optional[GraphRetrievalResult]) -> List[Dict[str, Any]]:
        """Extract supporting evidence from results."""
        evidence = []
        
        # Add vector evidence
        if vector_results:
            for result in vector_results.results[:3]:  # Top 3
                evidence.append({
                    'type': 'semantic',
                    'content': result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    'score': result.score,
                    'source': result.metadata.get('filename', 'Unknown document')
                })
        
        # Add graph evidence
        if graph_results:
            for rel in graph_results.relationships[:3]:  # Top 3
                subject = self._extract_entity_name(rel.get('subject', {}))
                predicate = self._extract_relation_type(rel.get('relationship', {}))
                obj = self._extract_entity_name(rel.get('object', {}))
                
                evidence.append({
                    'type': 'relational',
                    'content': f"{subject} {predicate} {obj}",
                    'confidence': rel.get('relationship', {}).get('confidence', 0.8),
                    'source': 'Knowledge graph'
                })
        
        return evidence
    
    def _generate_reasoning(self,
                           strategy: str,
                           vector_results: Optional[VectorRetrievalResult],
                           graph_results: Optional[GraphRetrievalResult]) -> str:
        """Generate reasoning explanation for the synthesis process."""
        reasoning_parts = []
        
        if strategy == "vector_only":
            reasoning_parts.append("Used semantic search to find relevant document content.")
            if vector_results:
                reasoning_parts.append(f"Found {len(vector_results.results)} relevant text segments.")
        
        elif strategy == "graph_only":
            reasoning_parts.append("Used graph traversal to find relevant entity relationships.")
            if graph_results:
                reasoning_parts.append(f"Found {len(graph_results.relationships)} relevant relationships.")
        
        elif strategy == "hybrid":
            reasoning_parts.append("Combined semantic search and graph traversal for comprehensive results.")
            if vector_results:
                reasoning_parts.append(f"Semantic search found {len(vector_results.results)} relevant segments.")
            if graph_results:
                reasoning_parts.append(f"Graph search found {len(graph_results.relationships)} relationships.")
        
        else:
            reasoning_parts.append("Used fallback approach due to insufficient data.")
        
        return " ".join(reasoning_parts)
    
    def _get_sources_summary(self,
                            vector_results: Optional[VectorRetrievalResult],
                            graph_results: Optional[GraphRetrievalResult]) -> Dict[str, Any]:
        """Get summary of sources used in synthesis."""
        summary = {
            'vector_search': False,
            'graph_search': False,
            'total_sources': 0
        }
        
        if vector_results and vector_results.results:
            summary['vector_search'] = True
            summary['vector_results_count'] = len(vector_results.results)
            summary['total_sources'] += len(set(
                r.metadata.get('filename', 'unknown') 
                for r in vector_results.results
            ))
        
        if graph_results and (graph_results.entities or graph_results.relationships):
            summary['graph_search'] = True
            summary['graph_entities_count'] = len(graph_results.entities)
            summary['graph_relationships_count'] = len(graph_results.relationships)
            summary['total_sources'] += 1  # Graph counts as one source
        
        return summary

if __name__ == "__main__":
    # Test synthesis agent
    from src.vector_store import SearchResult
    
    # Mock results for testing
    mock_vector_results = VectorRetrievalResult(
        results=[
            SearchResult(
                text="Microsoft reported strong revenue growth in Q3 2023, driven by cloud services.",
                score=0.89,
                metadata={'filename': 'microsoft_q3_2023.pdf'},
                chunk_id='chunk_1'
            )
        ],
        query="Microsoft financial performance",
        retrieval_time=0.15,
        metadata={'total_results': 1}
    )
    
    mock_graph_results = GraphRetrievalResult(
        entities=[{'name': 'Microsoft', 'type': 'Company'}],
        relationships=[{
            'subject': {'name': 'Satya Nadella', 'type': 'Person'},
            'relationship': {'type': 'CEO_OF', 'confidence': 0.95},
            'object': {'name': 'Microsoft', 'type': 'Company'}
        }],
        query="Microsoft leadership",
        query_type="executive",
        retrieval_time=0.12,
        metadata={'total_relationships': 1}
    )
    
    # Test synthesis
    synthesis_agent = SynthesisAgent()
    
    result = synthesis_agent.synthesize_answer(
        query="What is Microsoft's recent financial performance and who leads the company?",
        vector_results=mock_vector_results,
        graph_results=mock_graph_results
    )
    
    print("Synthesis Agent Test:")
    print(f"Query: What is Microsoft's recent financial performance and who leads the company?")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {result.sources_used}")
    print(f"Evidence: {len(result.supporting_evidence)} pieces")
    print(f"Synthesis time: {result.synthesis_time:.3f}s")
    
    print("\nSynthesis agent test completed successfully!")
