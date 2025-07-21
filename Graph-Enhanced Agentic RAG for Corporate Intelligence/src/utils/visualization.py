"""Visualization utilities for graphs and data."""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    plt = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create mock classes for type hints
    class MockFigure:
        pass
    go = MockFigure()
    go.Figure = MockFigure
    go.Scatter = MockFigure
    go.Bar = MockFigure
    go.Pie = MockFigure
    px = None
    make_subplots = None

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None

class GraphVisualizer:
    """Visualizes knowledge graphs and relationships."""
    
    def __init__(self):
        self.graph = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
    
    def add_entities_and_relations(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """Add entities and relations to the graph for visualization."""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, cannot create graph visualization")
            return
        
        # Clear existing graph
        self.graph.clear()
        
        # Add entities as nodes
        for entity in entities:
            entity_name = entity.get('name', str(entity))
            entity_type = entity.get('type', 'Unknown')
            
            self.graph.add_node(
                entity_name,
                type=entity_type,
                size=self._get_node_size_by_type(entity_type),
                color=self._get_node_color_by_type(entity_type)
            )
        
        # Add relations as edges
        for relation in relations:
            subject = self._extract_entity_name(relation.get('subject', {}))
            obj = self._extract_entity_name(relation.get('object', {}))
            rel_type = self._extract_relation_type(relation.get('relationship', {}))
            confidence = relation.get('relationship', {}).get('confidence', 1.0) if isinstance(relation.get('relationship'), dict) else 1.0
            
            if subject and obj:
                self.graph.add_edge(
                    subject, obj,
                    relation=rel_type,
                    weight=confidence,
                    color=self._get_edge_color_by_type(rel_type)
                )
    
    def create_interactive_graph(self, 
                                width: int = 800, 
                                height: int = 600,
                                output_path: str = "graph.html") -> Optional[str]:
        """Create an interactive graph visualization using pyvis."""
        if not PYVIS_AVAILABLE or not self.graph:
            logger.warning("Pyvis not available or no graph data, cannot create interactive visualization")
            return None
        
        try:
            # Create pyvis network
            net = Network(width=f"{width}px", height=f"{height}px", directed=True)
            
            # Configure physics
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Add nodes
            for node, attrs in self.graph.nodes(data=True):
                net.add_node(
                    node,
                    label=node,
                    color=attrs.get('color', '#97C2FC'),
                    size=attrs.get('size', 20),
                    title=f"Type: {attrs.get('type', 'Unknown')}"
                )
            
            # Add edges
            for source, target, attrs in self.graph.edges(data=True):
                net.add_edge(
                    source, target,
                    label=attrs.get('relation', ''),
                    color=attrs.get('color', '#848484'),
                    width=attrs.get('weight', 1) * 5,
                    title=f"Relation: {attrs.get('relation', 'Unknown')}"
                )
            
            # Save and return path
            net.save_graph(output_path)
            logger.info(f"Interactive graph saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create interactive graph: {e}")
            return None
    
    def create_static_graph(self, 
                           width: int = 12, 
                           height: int = 8,
                           output_path: Optional[str] = None) -> Optional[str]:
        """Create a static graph visualization using matplotlib."""
        if not NETWORKX_AVAILABLE or not plt or not self.graph:
            logger.warning("NetworkX/matplotlib not available or no graph data")
            return None
        
        try:
            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Draw nodes by type
            node_types = set(nx.get_node_attributes(self.graph, 'type').values())
            for node_type in node_types:
                nodes = [n for n, attrs in self.graph.nodes(data=True) if attrs.get('type') == node_type]
                if nodes:
                    nx.draw_networkx_nodes(
                        self.graph, pos,
                        nodelist=nodes,
                        node_color=self._get_node_color_by_type(node_type),
                        node_size=[self.graph.nodes[n].get('size', 300) for n in nodes],
                        alpha=0.8,
                        ax=ax
                    )
            
            # Draw edges by type
            edge_types = set(nx.get_edge_attributes(self.graph, 'relation').values())
            for edge_type in edge_types:
                edges = [(u, v) for u, v, attrs in self.graph.edges(data=True) if attrs.get('relation') == edge_type]
                if edges:
                    nx.draw_networkx_edges(
                        self.graph, pos,
                        edgelist=edges,
                        edge_color=self._get_edge_color_by_type(edge_type),
                        alpha=0.6,
                        arrows=True,
                        arrowsize=20,
                        ax=ax
                    )
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)
            
            # Add edge labels
            edge_labels = nx.get_edge_attributes(self.graph, 'relation')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6, ax=ax)
            
            ax.set_title("Knowledge Graph Visualization", fontsize=16)
            ax.axis('off')
            
            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Static graph saved to {output_path}")
            
            plt.tight_layout()
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create static graph: {e}")
            return None
    
    def create_plotly_graph(self) -> Optional[Any]:
        """Create an interactive graph using Plotly."""
        if not PLOTLY_AVAILABLE or not self.graph:
            logger.warning("Plotly not available or no graph data")
            return None
        
        try:
            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50) if NETWORKX_AVAILABLE else {}
            
            # Extract node information
            node_x = []
            node_y = []
            node_text = []
            node_hovertext = []
            
            # Extract edge information
            edge_trace = []
            
            for node in self.graph.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_info = f"{node}<br>Type: {self.graph.nodes[node].get('type', 'Unknown')}"
                    node_hovertext.append(node_info)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                text=node_text,
                hovertext=node_hovertext,
                mode='markers+text',
                hoverinfo='text',
                marker=dict(size=10, color='lightblue'),
                textposition="middle center"
            )
            
            # Add edges
            for edge in self.graph.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    edge_trace.append(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=2, color='#888'),
                            hoverinfo='none'
                        )
                    )
            
            # Create figure
            fig = go.Figure(data=[node_trace] + edge_trace,
                          layout=go.Layout(
                              title='Interactive Knowledge Graph',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Hover over nodes for more information",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color="#888", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create Plotly graph: {e}")
            return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph."""
        if not self.graph:
            return {'error': 'No graph data available'}
        
        try:
            stats = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if NETWORKX_AVAILABLE else 0,
                'node_types': {},
                'edge_types': {}
            }
            
            # Count node types
            for node, attrs in self.graph.nodes(data=True):
                node_type = attrs.get('type', 'Unknown')
                stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            # Count edge types
            for u, v, attrs in self.graph.edges(data=True):
                edge_type = attrs.get('relation', 'Unknown')
                stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {'error': str(e)}
    
    def _extract_entity_name(self, entity: Any) -> str:
        """Extract entity name from various formats."""
        if isinstance(entity, dict):
            return entity.get('name', str(entity))
        return str(entity)
    
    def _extract_relation_type(self, relationship: Any) -> str:
        """Extract relation type from various formats."""
        if isinstance(relationship, dict):
            return relationship.get('type', 'RELATED_TO')
        return str(relationship)
    
    def _get_node_size_by_type(self, node_type: str) -> int:
        """Get node size based on entity type."""
        size_map = {
            'Company': 500,
            'Person': 300,
            'Product': 250,
            'Financial': 200,
            'Location': 150,
            'Date': 100,
            'Unknown': 200
        }
        return size_map.get(node_type, 200)
    
    def _get_node_color_by_type(self, node_type: str) -> str:
        """Get node color based on entity type."""
        color_map = {
            'Company': '#FF6B6B',      # Red
            'Person': '#4ECDC4',       # Teal
            'Product': '#45B7D1',      # Blue
            'Financial': '#96CEB4',    # Green
            'Location': '#FFEAA7',     # Yellow
            'Date': '#DDA0DD',         # Plum
            'Unknown': '#95A5A6'       # Gray
        }
        return color_map.get(node_type, '#95A5A6')
    
    def _get_edge_color_by_type(self, edge_type: str) -> str:
        """Get edge color based on relation type."""
        color_map = {
            'CEO_OF': '#E74C3C',          # Red
            'ACQUIRED': '#3498DB',        # Blue
            'PARTNER_WITH': '#2ECC71',    # Green
            'OWNS': '#F39C12',            # Orange
            'SUBSIDIARY_OF': '#9B59B6',   # Purple
            'REPORTED': '#1ABC9C',        # Turquoise
            'Unknown': '#7F8C8D'          # Dark gray
        }
        return color_map.get(edge_type, '#7F8C8D')

class DataVisualizer:
    """Visualizes various data and statistics."""
    
    @staticmethod
    def create_retrieval_performance_chart(performance_data: Dict[str, Any]) -> Optional[Any]:
        """Create a chart showing retrieval performance metrics."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for performance visualization")
            return None
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Response Times', 'Result Counts', 'Confidence Scores', 'Source Distribution'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "pie"}]]
            )
            
            # Response times (if available)
            if 'response_times' in performance_data:
                times = performance_data['response_times']
                fig.add_trace(
                    go.Scatter(x=list(range(len(times))), y=times, mode='lines+markers', name='Response Time'),
                    row=1, col=1
                )
            
            # Result counts (if available)
            if 'result_counts' in performance_data:
                counts = performance_data['result_counts']
                fig.add_trace(
                    go.Bar(x=list(counts.keys()), y=list(counts.values()), name='Result Count'),
                    row=1, col=2
                )
            
            # Confidence scores (if available)
            if 'confidence_scores' in performance_data:
                scores = performance_data['confidence_scores']
                fig.add_trace(
                    go.Scatter(x=list(range(len(scores))), y=scores, mode='markers', name='Confidence'),
                    row=2, col=1
                )
            
            # Source distribution (if available)
            if 'source_distribution' in performance_data:
                sources = performance_data['source_distribution']
                fig.add_trace(
                    go.Pie(labels=list(sources.keys()), values=list(sources.values()), name='Sources'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=True, title_text="System Performance Dashboard")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create performance chart: {e}")
            return None
    
    @staticmethod
    def create_entity_distribution_chart(entity_counts: Dict[str, int]) -> Optional[Any]:
        """Create a chart showing entity type distribution."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for entity distribution visualization")
            return None
        
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(entity_counts.keys()),
                    y=list(entity_counts.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(entity_counts)]
                )
            ])
            
            fig.update_layout(
                title="Entity Type Distribution",
                xaxis_title="Entity Types",
                yaxis_title="Count",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create entity distribution chart: {e}")
            return None

if __name__ == "__main__":
    # Test visualization utilities
    print("Testing Visualization Utilities...")
    
    # Mock data for testing
    entities = [
        {'name': 'Microsoft', 'type': 'Company'},
        {'name': 'Satya Nadella', 'type': 'Person'},
        {'name': 'Azure', 'type': 'Product'},
        {'name': 'Apple', 'type': 'Company'},
        {'name': 'Tim Cook', 'type': 'Person'}
    ]
    
    relations = [
        {
            'subject': {'name': 'Satya Nadella', 'type': 'Person'},
            'relationship': {'type': 'CEO_OF', 'confidence': 0.95},
            'object': {'name': 'Microsoft', 'type': 'Company'}
        },
        {
            'subject': {'name': 'Microsoft', 'type': 'Company'},
            'relationship': {'type': 'OWNS', 'confidence': 0.9},
            'object': {'name': 'Azure', 'type': 'Product'}
        }
    ]
    
    # Test graph visualizer
    visualizer = GraphVisualizer()
    visualizer.add_entities_and_relations(entities, relations)
    
    # Get graph statistics
    stats = visualizer.get_graph_statistics()
    print(f"Graph statistics: {stats}")
    
    # Test creating visualizations (would save files if libraries available)
    interactive_path = visualizer.create_interactive_graph(output_path="test_graph.html")
    if interactive_path:
        print(f"Interactive graph would be saved to: {interactive_path}")
    
    static_path = visualizer.create_static_graph(output_path="test_graph.png")
    if static_path:
        print(f"Static graph would be saved to: {static_path}")
    
    # Test data visualizer
    data_viz = DataVisualizer()
    
    # Mock performance data
    performance_data = {
        'response_times': [0.15, 0.23, 0.18, 0.31, 0.19],
        'result_counts': {'Vector': 5, 'Graph': 3, 'Hybrid': 8},
        'confidence_scores': [0.85, 0.92, 0.78, 0.88, 0.91],
        'source_distribution': {'Document A': 15, 'Document B': 10, 'Graph DB': 8}
    }
    
    perf_fig = data_viz.create_retrieval_performance_chart(performance_data)
    if perf_fig:
        print("Performance chart created successfully")
    
    entity_counts = {'Company': 10, 'Person': 8, 'Product': 5, 'Financial': 3}
    entity_fig = data_viz.create_entity_distribution_chart(entity_counts)
    if entity_fig:
        print("Entity distribution chart created successfully")
    
    print("Visualization utilities test completed successfully!")
