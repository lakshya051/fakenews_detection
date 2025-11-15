"""
Network Construction Module for Misinformation Prediction.

This module handles building social networks from user interactions,
calculating network metrics, and detecting communities.
"""

import networkx as nx
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    logging.warning("python-igraph not available. Some features may be limited.")

import config

logging.basicConfig(**config.LOG_CONFIG)
logger = logging.getLogger(__name__)


def build_interaction_graph(df: pd.DataFrame,
                           user_column: str = "user_id",
                           interaction_column: str = "interaction_type",
                           target_column: Optional[str] = "target_user_id",
                           timestamp_column: Optional[str] = "timestamp",
                           weight_column: Optional[str] = None,
                           directed: bool = True) -> nx.DiGraph:
    """
    Build a directed graph from user interactions (retweets, replies, mentions).
    
    Args:
        df: DataFrame with interaction data
        user_column: Column name for source user
        interaction_column: Column name for interaction type
        target_column: Column name for target user (for directed edges)
        timestamp_column: Column name for timestamp
        weight_column: Column name for edge weights
        directed: Whether to create directed graph
        
    Returns:
        NetworkX graph object
        
    Example:
        >>> G = build_interaction_graph(df, "user_id", "interaction_type", "target_user")
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    logger.info(f"Building {'directed' if directed else 'undirected'} graph from interactions...")
    
    # Build edges from interactions
    edge_weights = defaultdict(int)
    
    for _, row in df.iterrows():
        source = row[user_column]
        
        # Handle different interaction types
        if target_column and pd.notna(row.get(target_column)):
            target = row[target_column]
        elif interaction_column in row:
            # Extract mentions from text
            if "@" in str(row.get("text", "")):
                mentions = extract_mentions(str(row.get("text", "")))
                for mention in mentions:
                    target = mention
                    weight = 1 if weight_column is None else row.get(weight_column, 1)
                    edge_weights[(source, target)] += weight
            continue
        else:
            continue
        
        weight = 1 if weight_column is None else row.get(weight_column, 1)
        edge_weights[(source, target)] += weight
    
    # Add edges to graph
    for (source, target), weight in edge_weights.items():
        G.add_edge(source, target, weight=weight)
    
    # Add node attributes (convert timestamps to strings to avoid GraphML issues)
    if timestamp_column and timestamp_column in df.columns:
        node_timestamps = df.groupby(user_column)[timestamp_column].first().to_dict()
        # Convert timestamps to strings for compatibility
        node_timestamps_str = {k: str(v) if pd.notna(v) else "" for k, v in node_timestamps.items()}
        nx.set_node_attributes(G, node_timestamps_str, "first_seen")
    
    logger.info(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    import re
    mentions = re.findall(r'@(\w+)', text)
    return mentions


def build_retweet_network(df: pd.DataFrame,
                         user_column: str = "user_id",
                         original_user_column: str = "original_user_id",
                         weight_column: str = "retweet_count") -> nx.DiGraph:
    """
    Build network from retweet relationships.
    
    Args:
        df: DataFrame with retweet data
        user_column: Column for user who retweeted
        original_user_column: Column for original tweet author
        weight_column: Column for retweet count/weight
        
    Returns:
        NetworkX directed graph
        
    Example:
        >>> G = build_retweet_network(df, "retweeter", "original_author")
    """
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        retweeter = row[user_column]
        original = row.get(original_user_column, row.get("original_user", None))
        
        if pd.notna(original) and retweeter != original:
            weight = row.get(weight_column, 1)
            if G.has_edge(retweeter, original):
                G[retweeter][original]['weight'] += weight
            else:
                G.add_edge(retweeter, original, weight=weight)
    
    logger.info(f"Retweet network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def calculate_network_statistics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculate basic network statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of network statistics
        
    Example:
        >>> stats = calculate_network_statistics(G)
    """
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": G.is_directed(),
        "is_connected": nx.is_connected(G.to_undirected()) if G.number_of_nodes() > 0 else False
    }
    
    if G.number_of_nodes() > 0:
        if G.is_directed():
            stats["avg_in_degree"] = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
            stats["avg_out_degree"] = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
        else:
            degrees = dict(G.degree())
            stats["avg_degree"] = sum(degrees.values()) / G.number_of_nodes()
        
        # Clustering coefficient
        if not G.is_directed():
            stats["avg_clustering"] = nx.average_clustering(G)
        else:
            stats["avg_clustering"] = nx.average_clustering(G.to_undirected())
    
    logger.info(f"Network statistics calculated: {stats}")
    return stats


def calculate_centrality_measures(G: nx.Graph,
                                  include_betweenness: bool = True,
                                  include_closeness: bool = True,
                                  include_eigenvector: bool = True) -> pd.DataFrame:
    """
    Calculate various centrality measures for nodes.
    
    Args:
        G: NetworkX graph
        include_betweenness: Whether to calculate betweenness centrality
        include_closeness: Whether to calculate closeness centrality
        include_eigenvector: Whether to calculate eigenvector centrality
        
    Returns:
        DataFrame with centrality scores for each node
        
    Example:
        >>> centrality_df = calculate_centrality_measures(G)
    """
    logger.info("Calculating centrality measures...")
    
    results = {}
    nodes = list(G.nodes())
    
    # Degree centrality
    if G.is_directed():
        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        results["in_degree_centrality"] = [in_degree.get(node, 0) for node in nodes]
        results["out_degree_centrality"] = [out_degree.get(node, 0) for node in nodes]
    else:
        degree = nx.degree_centrality(G)
        results["degree_centrality"] = [degree.get(node, 0) for node in nodes]
    
    # Betweenness centrality
    if include_betweenness:
        try:
            betweenness = nx.betweenness_centrality(G)
            results["betweenness_centrality"] = [betweenness.get(node, 0) for node in nodes]
        except Exception as e:
            logger.warning(f"Error calculating betweenness centrality: {e}")
            results["betweenness_centrality"] = [0] * len(nodes)
    
    # Closeness centrality
    if include_closeness:
        try:
            closeness = nx.closeness_centrality(G)
            results["closeness_centrality"] = [closeness.get(node, 0) for node in nodes]
        except Exception as e:
            logger.warning(f"Error calculating closeness centrality: {e}")
            results["closeness_centrality"] = [0] * len(nodes)
    
    # Eigenvector centrality
    if include_eigenvector:
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            results["eigenvector_centrality"] = [eigenvector.get(node, 0) for node in nodes]
        except Exception as e:
            logger.warning(f"Error calculating eigenvector centrality: {e}")
            results["eigenvector_centrality"] = [0] * len(nodes)
    
    # K-core decomposition
    try:
        k_core = nx.core_number(G.to_undirected() if G.is_directed() else G)
        results["k_core"] = [k_core.get(node, 0) for node in nodes]
    except Exception as e:
        logger.warning(f"Error calculating k-core: {e}")
        results["k_core"] = [0] * len(nodes)
    
    df = pd.DataFrame(results, index=nodes)
    logger.info(f"Calculated centrality measures for {len(nodes)} nodes")
    
    return df


def detect_communities(G: nx.Graph,
                      algorithm: str = "louvain") -> Dict[str, int]:
    """
    Detect communities in the network using specified algorithm.
    
    Args:
        G: NetworkX graph
        algorithm: Community detection algorithm ('louvain', 'leiden', 'greedy')
        
    Returns:
        Dictionary mapping node to community ID
        
    Example:
        >>> communities = detect_communities(G, "louvain")
    """
    logger.info(f"Detecting communities using {algorithm} algorithm...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    if algorithm == "louvain":
        try:
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(G_undirected)
        except ImportError:
            logger.warning("python-louvain not available. Using greedy modularity instead.")
            communities = nx.community.greedy_modularity_communities(G_undirected)
            # Convert to dict format
            community_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_dict[node] = i
            communities = community_dict
    elif algorithm == "leiden" and HAS_IGRAPH:
        # Convert to igraph
        ig_graph = ig.Graph.from_networkx(G_undirected)
        communities_ig = ig_graph.community_leiden()
        communities = {node: membership for node, membership in 
                      zip(G_undirected.nodes(), communities_ig.membership)}
    else:
        # Greedy modularity communities
        communities_list = nx.community.greedy_modularity_communities(G_undirected)
        communities = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                communities[node] = i
    
    logger.info(f"Detected {len(set(communities.values()))} communities")
    return communities


def identify_information_cascades(df: pd.DataFrame,
                                 user_column: str = "user_id",
                                 timestamp_column: str = "timestamp",
                                 post_id_column: str = "post_id") -> Dict[str, List[Dict]]:
    """
    Identify information cascades from propagation data.
    
    Args:
        df: DataFrame with propagation data
        user_column: Column for user IDs
        timestamp_column: Column for timestamps
        post_id_column: Column for post IDs
        
    Returns:
        Dictionary mapping post_id to cascade information
        
    Example:
        >>> cascades = identify_information_cascades(df)
    """
    logger.info("Identifying information cascades...")
    
    cascades = {}
    
    for post_id, group in df.groupby(post_id_column):
        cascade_data = group.sort_values(timestamp_column)
        
        # Calculate cascade metrics
        cascade_info = {
            "depth": calculate_cascade_depth(cascade_data, user_column, timestamp_column),
            "breadth": len(cascade_data),
            "users": cascade_data[user_column].tolist(),
            "timestamps": cascade_data[timestamp_column].tolist(),
            "propagation_speed": calculate_propagation_speed(cascade_data, timestamp_column)
        }
        
        cascades[post_id] = cascade_info
    
    logger.info(f"Identified {len(cascades)} information cascades")
    return cascades


def calculate_cascade_depth(cascade_df: pd.DataFrame,
                           user_column: str,
                           timestamp_column: str) -> int:
    """Calculate maximum depth of information cascade."""
    # Simple depth calculation based on time intervals
    if len(cascade_df) <= 1:
        return 0
    
    timestamps = pd.to_datetime(cascade_df[timestamp_column])
    time_diffs = timestamps.diff().dt.total_seconds()
    
    # Depth based on significant time gaps
    depth = 0
    threshold = 3600  # 1 hour threshold
    
    for diff in time_diffs[1:]:
        if diff > threshold:
            depth += 1
    
    return depth


def calculate_propagation_speed(cascade_df: pd.DataFrame,
                               timestamp_column: str) -> float:
    """Calculate average propagation speed (posts per hour)."""
    if len(cascade_df) <= 1:
        return 0.0
    
    timestamps = pd.to_datetime(cascade_df[timestamp_column])
    time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # hours
    
    if time_span == 0:
        return float('inf')
    
    return len(cascade_df) / time_span


def create_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Create adjacency matrix from graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        NumPy adjacency matrix
        
    Example:
        >>> adj_matrix = create_adjacency_matrix(G)
    """
    return nx.adjacency_matrix(G, nodelist=list(G.nodes())).toarray()


def export_network(G: nx.Graph,
                  output_path: Path,
                  format: str = "graphml") -> None:
    """
    Export network to file in various formats.
    
    Args:
        G: NetworkX graph
        output_path: Path to save the network
        format: Export format ('graphml', 'gml', 'json', 'edgelist')
        
    Example:
        >>> export_network(G, Path("data/networks/network.graphml"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a copy of the graph to avoid modifying the original
    G_export = G.copy()
    
    # Convert pandas Timestamps and other unsupported types to strings
    # for GraphML compatibility
    if format == "graphml":
        # Convert node attributes
        for node in G_export.nodes():
            node_attrs = G_export.nodes[node]
            for key, value in list(node_attrs.items()):
                # Check if value is a pandas Timestamp or similar datetime type
                try:
                    if pd.isna(value):
                        G_export.nodes[node][key] = ""
                        continue
                    # Check for Timestamp types
                    type_str = str(type(value))
                    if 'Timestamp' in type_str or 'datetime' in type_str:
                        G_export.nodes[node][key] = str(value)
                    elif isinstance(value, pd.Timestamp):
                        G_export.nodes[node][key] = str(value)
                except Exception:
                    # If conversion fails, try to convert to string anyway
                    try:
                        G_export.nodes[node][key] = str(value)
                    except:
                        # Remove attribute if can't convert
                        del G_export.nodes[node][key]
        
        # Convert edge attributes
        for u, v in G_export.edges():
            edge_attrs = G_export.edges[u, v]
            for key, value in list(edge_attrs.items()):
                try:
                    if pd.isna(value):
                        G_export.edges[u, v][key] = ""
                        continue
                    # Check for Timestamp types
                    type_str = str(type(value))
                    if 'Timestamp' in type_str or 'datetime' in type_str:
                        G_export.edges[u, v][key] = str(value)
                    elif isinstance(value, pd.Timestamp):
                        G_export.edges[u, v][key] = str(value)
                except Exception:
                    # If conversion fails, try to convert to string anyway
                    try:
                        G_export.edges[u, v][key] = str(value)
                    except:
                        # Remove attribute if can't convert
                        if key in G_export.edges[u, v]:
                            del G_export.edges[u, v][key]
    
    if format == "graphml":
        nx.write_graphml(G_export, output_path)
    elif format == "gml":
        nx.write_gml(G_export, output_path)
    elif format == "json":
        data = nx.node_link_data(G_export)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles timestamps
    elif format == "edgelist":
        nx.write_edgelist(G_export, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Exported network to {output_path} in {format} format")


def load_network(file_path: Path,
                format: str = "graphml") -> nx.Graph:
    """
    Load network from file.
    
    Args:
        file_path: Path to network file
        format: File format ('graphml', 'gml', 'json', 'edgelist')
        
    Returns:
        NetworkX graph
        
    Example:
        >>> G = load_network(Path("data/networks/network.graphml"))
    """
    if format == "graphml":
        G = nx.read_graphml(file_path)
    elif format == "gml":
        G = nx.read_gml(file_path)
    elif format == "json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
    elif format == "edgelist":
        G = nx.read_edgelist(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded network from {file_path}")
    return G

