"""
Causal Graph Builder for Blockchain Transaction Analysis
Defines causal relationships between transaction features and fraud outcomes
"""

import networkx as nx
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalGraphBuilder:
    """
    Builds a Directed Acyclic Graph (DAG) representing causal relationships
    in blockchain transactions for fraud detection
    """
    
    def __init__(self):
        """Initialize the causal graph structure"""
        self.graph = nx.DiGraph()
        self._build_domain_knowledge_graph()
    
    def _build_domain_knowledge_graph(self):
        """
        Define causal relationships based on blockchain domain knowledge
        
        Causal Structure:
        - Confounders: Variables that affect both features and outcome
        - Direct causes: Features that directly cause malicious behavior
        - Mediators: Variables that transmit causal effects
        """
        
        # Define nodes
        nodes = [
            'gas_price',
            'value',
            'gas_used',
            'sender_tx_count',
            'contract_age',
            'is_contract_creation',
            'gas_price_deviation',
            'block_gas_used_ratio',
            'function_signature_hash',
            'malicious',  # Outcome variable
            # Latent/confounding variables
            'network_congestion',  # Confounder
            'sender_intent',       # Confounder
            'contract_complexity'  # Confounder
        ]
        
        self.graph.add_nodes_from(nodes)
        
        # Define causal edges (cause -> effect)
        # Format: (from, to, causal_mechanism)
        causal_edges = [
            # Network congestion affects gas prices and fraud detection
            ('network_congestion', 'gas_price', 'congestion_drives_price'),
            ('network_congestion', 'gas_price_deviation', 'congestion_variance'),
            ('network_congestion', 'block_gas_used_ratio', 'congestion_blocks'),
            
            # Sender intent influences multiple features
            ('sender_intent', 'gas_price', 'intent_pricing'),
            ('sender_intent', 'value', 'intent_amount'),
            ('sender_intent', 'malicious', 'intent_fraud'),
            
            # Contract complexity affects execution
            ('contract_complexity', 'gas_used', 'complexity_gas'),
            ('contract_complexity', 'function_signature_hash', 'complexity_functions'),
            
            # Direct causal relationships to malicious outcome
            ('gas_price_deviation', 'malicious', 'abnormal_pricing_fraud'),
            ('sender_tx_count', 'malicious', 'new_account_risk'),
            ('is_contract_creation', 'malicious', 'creation_risk'),
            ('value', 'malicious', 'high_value_fraud'),
            
            # Mediating relationships
            ('gas_price', 'gas_price_deviation', 'price_variance'),
            ('contract_age', 'sender_tx_count', 'age_activity'),
            ('gas_used', 'block_gas_used_ratio', 'usage_ratio'),
        ]
        
        for source, target, mechanism in causal_edges:
            self.graph.add_edge(source, target, mechanism=mechanism)
        
        logger.info(f"Built causal graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def get_graph(self) -> nx.DiGraph:
        """Return the causal graph"""
        return self.graph
    
    def get_parents(self, node: str) -> List[str]:
        """Get direct causes (parents) of a node"""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get direct effects (children) of a node"""
        return list(self.graph.successors(node))
    
    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """
        Identify confounders: variables that affect both treatment and outcome
        
        A confounder must:
        1. Be a common cause of both treatment and outcome
        2. Not be a descendant of treatment
        """
        confounders = []
        
        for node in self.graph.nodes():
            if node == treatment or node == outcome:
                continue
            
            # Check if node affects both treatment and outcome
            affects_treatment = nx.has_path(self.graph, node, treatment)
            affects_outcome = nx.has_path(self.graph, node, outcome)
            
            # Check if node is not a descendant of treatment
            not_descendant = not nx.has_path(self.graph, treatment, node)
            
            if affects_treatment and affects_outcome and not_descendant:
                confounders.append(node)
        
        return confounders
    
    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """
        Identify mediators: variables on the causal path from treatment to outcome
        """
        mediators = []
        
        try:
            # Get all simple paths from treatment to outcome
            paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
            
            for path in paths:
                # All nodes between treatment and outcome are mediators
                mediators.extend(path[1:-1])
            
            return list(set(mediators))  # Remove duplicates
        except nx.NetworkXNoPath:
            return []
    
    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Find backdoor paths that create spurious correlations
        Backdoor path: treatment <- confounder -> outcome
        """
        backdoor_paths = []
        
        # Create undirected version of graph
        undirected = self.graph.to_undirected()
        
        # Find all paths in undirected graph
        try:
            all_paths = list(nx.all_simple_paths(undirected, treatment, outcome))
            
            for path in all_paths:
                # A backdoor path has an arrow INTO treatment
                if len(path) > 2:  # At least one intermediate node
                    # Check if first edge goes into treatment
                    second_node = path[1]
                    if self.graph.has_edge(second_node, treatment):
                        backdoor_paths.append(path)
        except nx.NetworkXNoPath:
            pass
        
        return backdoor_paths
    
    def get_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Get the minimal set of variables to control for to estimate causal effect
        This blocks all backdoor paths
        """
        confounders = self.get_confounders(treatment, outcome)
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)
        
        # Collect all nodes on backdoor paths (excluding treatment and outcome)
        adjustment_set = set()
        for path in backdoor_paths:
            adjustment_set.update([node for node in path if node not in [treatment, outcome]])
        
        # Add identified confounders
        adjustment_set.update(confounders)
        
        return list(adjustment_set)
    
    def get_frontdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Find frontdoor paths for causal effect estimation
        Frontdoor path: treatment -> mediator -> outcome
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def visualize_structure(self) -> Dict:
        """
        Return graph structure for visualization
        """
        nodes = []
        edges = []
        
        for node in self.graph.nodes():
            node_type = 'outcome' if node == 'malicious' else \
                       'confounder' if node in ['network_congestion', 'sender_intent', 'contract_complexity'] else \
                       'feature'
            
            nodes.append({
                'id': node,
                'label': node.replace('_', ' ').title(),
                'type': node_type
            })
        
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'from': source,
                'to': target,
                'mechanism': data.get('mechanism', 'unknown')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes())
            }
        }
