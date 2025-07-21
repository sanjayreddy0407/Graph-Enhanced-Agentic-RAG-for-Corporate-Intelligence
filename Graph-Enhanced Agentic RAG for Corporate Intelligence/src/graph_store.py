"""Graph database operations using Neo4j for relational queries."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

from src.knowledge_extractor import Entity, Relation

logger = logging.getLogger(__name__)

@dataclass
class GraphQueryResult:
    """Result from a graph database query."""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class GraphStore:
    """Neo4j-based graph store for relational queries."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database."""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available, using mock implementation")
            return
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.warning("Using mock graph store implementation")
            self.driver = None
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)."""
        if self.driver is None:
            logger.warning("No database connection, skipping clear")
            return
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def add_entities(self, entities: List[Entity]):
        """Add entities as nodes to the graph."""
        if self.driver is None:
            logger.warning("No database connection, using mock storage")
            self._mock_add_entities(entities)
            return
        
        with self.driver.session() as session:
            for entity in entities:
                # Create node with entity type as label
                cypher = f"""
                MERGE (e:{entity.type.capitalize()} {{name: $name}})
                SET e.confidence = $confidence,
                    e.context = $context
                RETURN e
                """
                
                session.run(cypher, {
                    'name': entity.name,
                    'confidence': entity.confidence,
                    'context': entity.context
                })
        
        logger.info(f"Added {len(entities)} entities to graph")
    
    def add_relations(self, relations: List[Relation]):
        """Add relations as edges to the graph."""
        if self.driver is None:
            logger.warning("No database connection, using mock storage")
            self._mock_add_relations(relations)
            return
        
        with self.driver.session() as session:
            for relation in relations:
                # Create relationship between nodes
                cypher = """
                MATCH (s) WHERE s.name = $subject
                MATCH (o) WHERE o.name = $object
                MERGE (s)-[r:RELATES {type: $predicate}]->(o)
                SET r.confidence = $confidence,
                    r.context = $context
                RETURN r
                """
                
                session.run(cypher, {
                    'subject': relation.subject,
                    'object': relation.object,
                    'predicate': relation.predicate,
                    'confidence': relation.confidence,
                    'context': relation.context
                })
        
        logger.info(f"Added {len(relations)} relations to graph")
    
    def find_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Find an entity by name."""
        if self.driver is None:
            return self._mock_find_entity(entity_name)
        
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e) WHERE e.name = $name RETURN e",
                {'name': entity_name}
            )
            
            record = result.single()
            if record:
                node = record['e']
                return dict(node)
            return None
    
    def find_relationships(self, entity_name: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find relationships for an entity."""
        if self.driver is None:
            return self._mock_find_relationships(entity_name, relation_type)
        
        with self.driver.session() as session:
            if relation_type:
                cypher = """
                MATCH (s)-[r:RELATES]->(o) 
                WHERE (s.name = $name OR o.name = $name) AND r.type = $relation_type
                RETURN s, r, o
                """
                params = {'name': entity_name, 'relation_type': relation_type}
            else:
                cypher = """
                MATCH (s)-[r:RELATES]->(o) 
                WHERE s.name = $name OR o.name = $name
                RETURN s, r, o
                """
                params = {'name': entity_name}
            
            results = session.run(cypher, params)
            
            relationships = []
            for record in results:
                relationships.append({
                    'subject': dict(record['s']),
                    'relationship': dict(record['r']),
                    'object': dict(record['o'])
                })
            
            return relationships
    
    def find_connected_entities(self, entity_name: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Find entities connected to a given entity within max_hops."""
        if self.driver is None:
            return self._mock_find_connected_entities(entity_name, max_hops)
        
        with self.driver.session() as session:
            cypher = f"""
            MATCH path = (start)-[*1..{max_hops}]-(connected)
            WHERE start.name = $name
            RETURN DISTINCT connected, length(path) as distance
            ORDER BY distance
            """
            
            results = session.run(cypher, {'name': entity_name})
            
            connected_entities = []
            for record in results:
                entity_data = dict(record['connected'])
                entity_data['distance'] = record['distance']
                connected_entities.append(entity_data)
            
            return connected_entities
    
    def query_by_relation_type(self, relation_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query relationships by type (e.g., 'CEO_OF', 'ACQUIRED')."""
        if self.driver is None:
            return self._mock_query_by_relation_type(relation_type, limit)
        
        with self.driver.session() as session:
            cypher = """
            MATCH (s)-[r:RELATES]->(o) 
            WHERE r.type = $relation_type
            RETURN s, r, o
            ORDER BY r.confidence DESC
            LIMIT $limit
            """
            
            results = session.run(cypher, {'relation_type': relation_type, 'limit': limit})
            
            relationships = []
            for record in results:
                relationships.append({
                    'subject': dict(record['s']),
                    'relationship': dict(record['r']),
                    'object': dict(record['o'])
                })
            
            return relationships
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        if self.driver is None:
            return self._mock_get_statistics()
        
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = {}
            node_result = session.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as count"
            )
            for record in node_result:
                node_counts[record['label']] = record['count']
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relationship_count = rel_result.single()['count']
            
            return {
                'node_counts': node_counts,
                'total_relationships': relationship_count,
                'total_nodes': sum(node_counts.values())
            }
    
    def execute_cypher_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query."""
        if self.driver is None:
            logger.warning("No database connection, returning empty results")
            return []
        
        if parameters is None:
            parameters = {}
        
        with self.driver.session() as session:
            results = session.run(query, parameters)
            return [dict(record) for record in results]
    
    # Mock implementations for development/testing
    def _mock_add_entities(self, entities: List[Entity]):
        """Mock implementation for adding entities."""
        logger.info(f"Mock: Added {len(entities)} entities")
    
    def _mock_add_relations(self, relations: List[Relation]):
        """Mock implementation for adding relations."""
        logger.info(f"Mock: Added {len(relations)} relations")
    
    def _mock_find_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Mock implementation for finding entity."""
        # Return sample entity data
        return {
            'name': entity_name,
            'type': 'Company',
            'confidence': 0.8
        }
    
    def _mock_find_relationships(self, entity_name: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Mock implementation for finding relationships."""
        # Return sample relationships
        return [
            {
                'subject': {'name': entity_name, 'type': 'Company'},
                'relationship': {'type': 'CEO_OF', 'confidence': 0.9},
                'object': {'name': 'John Doe', 'type': 'Person'}
            }
        ]
    
    def _mock_find_connected_entities(self, entity_name: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Mock implementation for finding connected entities."""
        return [
            {'name': 'Connected Entity 1', 'type': 'Company', 'distance': 1},
            {'name': 'Connected Entity 2', 'type': 'Person', 'distance': 2}
        ]
    
    def _mock_query_by_relation_type(self, relation_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock implementation for querying by relation type."""
        return [
            {
                'subject': {'name': 'Company A', 'type': 'Company'},
                'relationship': {'type': relation_type, 'confidence': 0.85},
                'object': {'name': 'Company B', 'type': 'Company'}
            }
        ]
    
    def _mock_get_statistics(self) -> Dict[str, Any]:
        """Mock implementation for getting statistics."""
        return {
            'node_counts': {'Company': 10, 'Person': 5, 'Product': 3},
            'total_relationships': 15,
            'total_nodes': 18
        }

class GraphQueryBuilder:
    """Helper class for building Cypher queries."""
    
    @staticmethod
    def find_acquisitions(company_name: str) -> str:
        """Build query to find acquisitions by a company."""
        return """
        MATCH (acquirer)-[r:RELATES]->(acquired)
        WHERE acquirer.name = $company_name AND r.type IN ['ACQUIRED', 'BOUGHT', 'PURCHASED']
        RETURN acquirer, r, acquired
        ORDER BY r.confidence DESC
        """
    
    @staticmethod
    def find_executives(company_name: str) -> str:
        """Build query to find executives of a company."""
        return """
        MATCH (person)-[r:RELATES]->(company)
        WHERE company.name = $company_name AND r.type IN ['CEO_OF', 'EXECUTIVE_OF', 'PRESIDENT_OF']
        RETURN person, r, company
        ORDER BY r.confidence DESC
        """
    
    @staticmethod
    def find_partnerships(company_name: str) -> str:
        """Build query to find partnerships of a company."""
        return """
        MATCH (company1)-[r:RELATES]-(company2)
        WHERE (company1.name = $company_name OR company2.name = $company_name) 
        AND r.type IN ['PARTNER_WITH', 'COLLABORATED_WITH', 'JOINT_VENTURE']
        RETURN company1, r, company2
        ORDER BY r.confidence DESC
        """

if __name__ == "__main__":
    # Example usage
    from config import Config
    
    graph_store = GraphStore(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD
    )
    
    # Sample entities and relations would be added here
    print("Graph store initialized successfully!")
    
    # Test statistics
    stats = graph_store.get_graph_statistics()
    print(f"Graph statistics: {stats}")
    
    graph_store.close()
