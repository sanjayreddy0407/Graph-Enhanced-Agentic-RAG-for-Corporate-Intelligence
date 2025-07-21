"""Graph retrieval agent for relationship-based queries."""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import re

from src.graph_store import GraphStore, GraphQueryBuilder

logger = logging.getLogger(__name__)

@dataclass
class GraphRetrievalResult:
    """Result from graph-based retrieval."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query: str
    query_type: str
    retrieval_time: float
    metadata: Dict[str, Any]

class GraphAgent:
    """Agent responsible for graph-based relational retrieval."""
    
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.query_builder = GraphQueryBuilder()
        self.max_results = 20
    
    def retrieve(self, query: str, query_context: Dict[str, Any] = None) -> GraphRetrievalResult:
        """
        Perform graph-based retrieval for relational queries.
        
        Args:
            query: User query
            query_context: Additional context about the query
            
        Returns:
            GraphRetrievalResult with entities and relationships
        """
        import time
        start_time = time.time()
        
        logger.info(f"Graph retrieval for query: '{query[:50]}...'")
        
        try:
            # Analyze query to determine graph retrieval strategy
            query_analysis = self._analyze_query(query)
            
            entities = []
            relationships = []
            
            # Execute appropriate graph queries based on analysis
            if query_analysis['type'] == 'acquisition':
                entities, relationships = self._handle_acquisition_query(query, query_analysis)
            elif query_analysis['type'] == 'executive':
                entities, relationships = self._handle_executive_query(query, query_analysis)
            elif query_analysis['type'] == 'partnership':
                entities, relationships = self._handle_partnership_query(query, query_analysis)
            elif query_analysis['type'] == 'ownership':
                entities, relationships = self._handle_ownership_query(query, query_analysis)
            else:
                # General relationship search
                entities, relationships = self._handle_general_query(query, query_analysis)
            
            retrieval_time = time.time() - start_time
            
            logger.info(f"Graph retrieval completed: {len(entities)} entities, {len(relationships)} relationships in {retrieval_time:.3f}s")
            
            return GraphRetrievalResult(
                entities=entities,
                relationships=relationships,
                query=query,
                query_type=query_analysis['type'],
                retrieval_time=retrieval_time,
                metadata={
                    'analysis': query_analysis,
                    'total_entities': len(entities),
                    'total_relationships': len(relationships)
                }
            )
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return GraphRetrievalResult(
                entities=[],
                relationships=[],
                query=query,
                query_type='error',
                retrieval_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine the type of graph traversal needed."""
        query_lower = query.lower()
        
        # Detect query patterns
        patterns = {
            'acquisition': [
                r'(acquired|bought|purchased|acquisition)',
                r'(merger|merged with)',
                r'(owns|owned)',
                r'(subsidiary|subsidiaries)'
            ],
            'executive': [
                r'(ceo|chief executive)',
                r'(president|chairman)',
                r'(executive|management)',
                r'(founder|co-founder)',
                r'(director|board member)'
            ],
            'partnership': [
                r'(partner|partnership)',
                r'(collaboration|joint venture)',
                r'(alliance|cooperation)',
                r'(supplier|customer)'
            ],
            'ownership': [
                r'(owns|owned by)',
                r'(subsidiary|parent company)',
                r'(holding company|conglomerate)',
                r'(division|unit)'
            ]
        }
        
        # Find matching patterns
        detected_types = []
        for query_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, query_lower):
                    detected_types.append(query_type)
                    break
        
        # Extract potential entity names (simple heuristic)
        potential_entities = self._extract_entity_names(query)
        
        # Determine primary query type
        primary_type = detected_types[0] if detected_types else 'general'
        
        return {
            'type': primary_type,
            'detected_types': detected_types,
            'entities': potential_entities,
            'has_list_intent': any(word in query_lower for word in ['list', 'show', 'all', 'which']),
            'has_specific_entity': len(potential_entities) > 0
        }
    
    def _extract_entity_names(self, query: str) -> List[str]:
        """Extract potential entity names from the query."""
        # Look for capitalized words that might be company/person names
        import re
        
        # Pattern for potential company names
        company_patterns = [
            r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(?:Inc|Corp|Corporation|Ltd|Limited|Company|Co)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s+(?:acquired|bought|owns|CEO|president))'
        ]
        
        entities = []
        for pattern in company_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entity = match.group(1).strip()
                if entity not in entities and len(entity) > 2:
                    entities.append(entity)
        
        return entities
    
    def _handle_acquisition_query(self, query: str, analysis: Dict[str, Any]) -> tuple:
        """Handle acquisition-related queries."""
        entities = []
        relationships = []
        
        if analysis['entities']:
            # Query for specific company's acquisitions
            company_name = analysis['entities'][0]
            cypher_query = self.query_builder.find_acquisitions(company_name)
            results = self.graph_store.execute_cypher_query(cypher_query, {'company_name': company_name})
            
            for result in results:
                if 'acquirer' in result:
                    entities.append(result['acquirer'])
                if 'acquired' in result:
                    entities.append(result['acquired'])
                if 'r' in result:
                    relationships.append({
                        'subject': result['acquirer'],
                        'relationship': result['r'],
                        'object': result['acquired']
                    })
        else:
            # General acquisition query
            results = self.graph_store.query_by_relation_type('ACQUIRED', limit=self.max_results)
            for result in results:
                entities.extend([result['subject'], result['object']])
                relationships.append(result)
        
        return self._deduplicate_results(entities, relationships)
    
    def _handle_executive_query(self, query: str, analysis: Dict[str, Any]) -> tuple:
        """Handle executive/leadership-related queries."""
        entities = []
        relationships = []
        
        if analysis['entities']:
            # Query for specific company's executives
            company_name = analysis['entities'][0]
            cypher_query = self.query_builder.find_executives(company_name)
            results = self.graph_store.execute_cypher_query(cypher_query, {'company_name': company_name})
            
            for result in results:
                if 'person' in result:
                    entities.append(result['person'])
                if 'company' in result:
                    entities.append(result['company'])
                if 'r' in result:
                    relationships.append({
                        'subject': result['person'],
                        'relationship': result['r'],
                        'object': result['company']
                    })
        else:
            # General executive relationships
            executive_relations = ['CEO_OF', 'EXECUTIVE_OF', 'PRESIDENT_OF']
            for relation in executive_relations:
                results = self.graph_store.query_by_relation_type(relation, limit=10)
                for result in results:
                    entities.extend([result['subject'], result['object']])
                    relationships.append(result)
        
        return self._deduplicate_results(entities, relationships)
    
    def _handle_partnership_query(self, query: str, analysis: Dict[str, Any]) -> tuple:
        """Handle partnership-related queries."""
        entities = []
        relationships = []
        
        if analysis['entities']:
            # Query for specific company's partnerships
            company_name = analysis['entities'][0]
            cypher_query = self.query_builder.find_partnerships(company_name)
            results = self.graph_store.execute_cypher_query(cypher_query, {'company_name': company_name})
            
            for result in results:
                if 'company1' in result:
                    entities.append(result['company1'])
                if 'company2' in result:
                    entities.append(result['company2'])
                if 'r' in result:
                    relationships.append({
                        'subject': result['company1'],
                        'relationship': result['r'],
                        'object': result['company2']
                    })
        else:
            # General partnership relationships
            partnership_relations = ['PARTNER_WITH', 'COLLABORATED_WITH', 'JOINT_VENTURE']
            for relation in partnership_relations:
                results = self.graph_store.query_by_relation_type(relation, limit=10)
                for result in results:
                    entities.extend([result['subject'], result['object']])
                    relationships.append(result)
        
        return self._deduplicate_results(entities, relationships)
    
    def _handle_ownership_query(self, query: str, analysis: Dict[str, Any]) -> tuple:
        """Handle ownership-related queries."""
        entities = []
        relationships = []
        
        # Look for ownership relationships
        ownership_relations = ['OWNS', 'SUBSIDIARY_OF', 'PARENT_OF']
        for relation in ownership_relations:
            results = self.graph_store.query_by_relation_type(relation, limit=self.max_results)
            for result in results:
                entities.extend([result['subject'], result['object']])
                relationships.append(result)
        
        return self._deduplicate_results(entities, relationships)
    
    def _handle_general_query(self, query: str, analysis: Dict[str, Any]) -> tuple:
        """Handle general relationship queries."""
        entities = []
        relationships = []
        
        if analysis['entities']:
            # Find all relationships for the mentioned entities
            for entity_name in analysis['entities']:
                entity_relationships = self.graph_store.find_relationships(entity_name)
                for rel in entity_relationships:
                    entities.extend([rel['subject'], rel['object']])
                    relationships.append(rel)
        else:
            # Return a sample of all relationships
            stats = self.graph_store.get_graph_statistics()
            if stats.get('total_relationships', 0) > 0:
                # Get relationships from different types
                sample_results = self.graph_store.execute_cypher_query(
                    "MATCH (s)-[r]->(o) RETURN s, r, o LIMIT $limit",
                    {'limit': min(self.max_results, 50)}
                )
                
                for result in sample_results:
                    if 's' in result and 'o' in result and 'r' in result:
                        entities.extend([result['s'], result['o']])
                        relationships.append({
                            'subject': result['s'],
                            'relationship': result['r'],
                            'object': result['o']
                        })
        
        return self._deduplicate_results(entities, relationships)
    
    def _deduplicate_results(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> tuple:
        """Remove duplicate entities and relationships."""
        # Deduplicate entities by name
        seen_entities = set()
        unique_entities = []
        for entity in entities:
            if isinstance(entity, dict) and 'name' in entity:
                if entity['name'] not in seen_entities:
                    seen_entities.add(entity['name'])
                    unique_entities.append(entity)
        
        # Deduplicate relationships
        seen_relationships = set()
        unique_relationships = []
        for rel in relationships:
            if isinstance(rel, dict):
                # Create a unique key for the relationship
                subject_name = rel.get('subject', {}).get('name', '') if isinstance(rel.get('subject'), dict) else str(rel.get('subject', ''))
                object_name = rel.get('object', {}).get('name', '') if isinstance(rel.get('object'), dict) else str(rel.get('object', ''))
                rel_type = rel.get('relationship', {}).get('type', '') if isinstance(rel.get('relationship'), dict) else ''
                
                rel_key = (subject_name, rel_type, object_name)
                if rel_key not in seen_relationships:
                    seen_relationships.add(rel_key)
                    unique_relationships.append(rel)
        
        return unique_entities, unique_relationships
    
    def find_entity_connections(self, entity_name: str, max_hops: int = 2) -> GraphRetrievalResult:
        """
        Find all entities connected to a given entity.
        
        Args:
            entity_name: Name of the entity to find connections for
            max_hops: Maximum number of relationship hops to traverse
            
        Returns:
            GraphRetrievalResult with connected entities
        """
        import time
        start_time = time.time()
        
        logger.info(f"Finding connections for entity: {entity_name}")
        
        try:
            connected_entities = self.graph_store.find_connected_entities(entity_name, max_hops)
            relationships = self.graph_store.find_relationships(entity_name)
            
            retrieval_time = time.time() - start_time
            
            return GraphRetrievalResult(
                entities=connected_entities,
                relationships=relationships,
                query=f"Find connections for {entity_name}",
                query_type="connection_search",
                retrieval_time=retrieval_time,
                metadata={
                    'target_entity': entity_name,
                    'max_hops': max_hops,
                    'total_entities': len(connected_entities),
                    'total_relationships': len(relationships)
                }
            )
            
        except Exception as e:
            logger.error(f"Connection search failed: {e}")
            return GraphRetrievalResult(
                entities=[],
                relationships=[],
                query=f"Find connections for {entity_name}",
                query_type="connection_search",
                retrieval_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def explain_retrieval(self, result: GraphRetrievalResult) -> Dict[str, Any]:
        """Provide explanation for graph retrieval results."""
        if not result.entities and not result.relationships:
            return {
                'explanation': 'No graph relationships found for the query',
                'suggestions': [
                    'Check if entities mentioned in the query exist in the graph',
                    'Try broader relationship terms',
                    'Ensure graph database contains relevant data'
                ]
            }
        
        # Analyze relationship types
        relationship_types = {}
        for rel in result.relationships:
            if isinstance(rel, dict) and 'relationship' in rel:
                rel_type = rel['relationship'].get('type', 'unknown') if isinstance(rel['relationship'], dict) else 'unknown'
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        # Analyze entity types
        entity_types = {}
        for entity in result.entities:
            if isinstance(entity, dict):
                entity_type = entity.get('type', 'unknown')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return {
            'total_entities': len(result.entities),
            'total_relationships': len(result.relationships),
            'query_type': result.query_type,
            'relationship_breakdown': relationship_types,
            'entity_breakdown': entity_types,
            'retrieval_time': round(result.retrieval_time, 3),
            'data_quality': self._assess_graph_result_quality(result)
        }
    
    def _assess_graph_result_quality(self, result: GraphRetrievalResult) -> str:
        """Assess the quality of graph retrieval results."""
        entity_count = len(result.entities)
        relationship_count = len(result.relationships)
        
        if entity_count == 0 and relationship_count == 0:
            return "No results found"
        elif relationship_count > entity_count * 0.5:
            return "High quality - rich relationship data"
        elif relationship_count > 0:
            return "Good quality - some relationship data found"
        else:
            return "Limited quality - entities found but few relationships"

if __name__ == "__main__":
    # Mock testing without actual graph store
    class MockGraphStore:
        def execute_cypher_query(self, query, params=None):
            return [
                {
                    'acquirer': {'name': 'Microsoft', 'type': 'Company'},
                    'r': {'type': 'ACQUIRED', 'confidence': 0.9},
                    'acquired': {'name': 'LinkedIn', 'type': 'Company'}
                }
            ]
        
        def query_by_relation_type(self, relation_type, limit=10):
            return [
                {
                    'subject': {'name': 'Satya Nadella', 'type': 'Person'},
                    'relationship': {'type': relation_type, 'confidence': 0.95},
                    'object': {'name': 'Microsoft', 'type': 'Company'}
                }
            ]
        
        def find_relationships(self, entity_name):
            return [
                {
                    'subject': {'name': entity_name, 'type': 'Company'},
                    'relationship': {'type': 'ACQUIRED', 'confidence': 0.8},
                    'object': {'name': 'Some Company', 'type': 'Company'}
                }
            ]
        
        def find_connected_entities(self, entity_name, max_hops):
            return [
                {'name': 'Connected Entity', 'type': 'Company', 'distance': 1}
            ]
        
        def get_graph_statistics(self):
            return {'total_relationships': 100, 'total_nodes': 50}
    
    # Test graph agent
    mock_store = MockGraphStore()
    agent = GraphAgent(mock_store)
    
    test_queries = [
        "Which companies did Microsoft acquire?",
        "Who is the CEO of Apple?",
        "List partnerships of Google"
    ]
    
    print("Graph Agent Test Results:")
    print("=" * 40)
    
    for query in test_queries:
        result = agent.retrieve(query)
        explanation = agent.explain_retrieval(result)
        
        print(f"\nQuery: {query}")
        print(f"Type: {result.query_type}")
        print(f"Entities: {len(result.entities)}")
        print(f"Relationships: {len(result.relationships)}")
        print(f"Quality: {explanation['data_quality']}")
        print(f"Time: {result.retrieval_time:.3f}s")
        print("-" * 30)
    
    print("Graph agent test completed successfully!")
