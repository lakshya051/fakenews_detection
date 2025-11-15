"""
Visualization Module for Misinformation Prediction.

This module provides functions for visualizing networks, features,
model performance, and propagation patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logging.warning("Plotly not available. Interactive visualizations disabled.")

import config

logging.basicConfig(**config.LOG_CONFIG)
logger = logging.getLogger(__name__)

# Set style
plt.style.use(config.VIZ_CONFIG.get("style", "seaborn-v0_8"))
sns.set_palette(config.VIZ_CONFIG.get("color_palette", "Set2"))


def plot_network_graph(G, 
                       node_colors: Optional[Dict] = None,
                       node_labels: Optional[Dict] = None,
                       edge_weights: bool = True,
                       layout: str = "spring",
                       figsize: Tuple[int, int] = (12, 8),
                       title: str = "Social Network Graph",
                       save_path: Optional[Path] = None) -> None:
    """
    Plot network graph with color-coded nodes.
    
    Args:
        G: NetworkX graph
        node_colors: Dictionary mapping nodes to colors
        node_labels: Dictionary mapping nodes to labels
        edge_weights: Whether to show edge weights
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        
    Example:
        >>> plot_network_graph(G, node_colors={node: 'red' if label==1 else 'blue'})
    """
    logger.info("Plotting network graph...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw edges
    if edge_weights and G.is_weighted():
        edges = G.edges(data=True)
        weights = [edge[2].get('weight', 1) for edge in edges]
        nx.draw_networkx_edges(G, pos, width=[w/10 for w in weights], 
                              alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    
    # Draw nodes
    if node_colors:
        node_color_list = [node_colors.get(node, 'gray') for node in G.nodes()]
    else:
        node_color_list = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                          node_size=500, alpha=0.8, ax=ax)
    
    # Draw labels
    if node_labels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved network plot to {save_path}")
    
    plt.show()


def plot_information_cascade(cascade_data: Dict,
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[Path] = None) -> None:
    """
    Visualize information diffusion cascade.
    
    Args:
        cascade_data: Dictionary with cascade information
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_information_cascade(cascade_info)
    """
    logger.info("Plotting information cascade...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Timeline plot
    timestamps = pd.to_datetime(cascade_data.get("timestamps", []))
    users = cascade_data.get("users", [])
    
    if len(timestamps) > 0:
        ax1.scatter(timestamps, range(len(timestamps)), alpha=0.6, s=100)
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("User Index", fontsize=12)
        ax1.set_title("Cascade Timeline", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Depth vs Breadth
    depth = cascade_data.get("depth", 0)
    breadth = cascade_data.get("breadth", 0)
    
    ax2.bar(["Depth", "Breadth"], [depth, breadth], color=['skyblue', 'lightcoral'])
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Cascade Metrics", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved cascade plot to {save_path}")
    
    plt.show()


def plot_feature_importance(model, 
                           feature_names: List[str],
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_feature_importance(rf_model, feature_names, top_n=15)
    """
    logger.info("Plotting feature importance...")
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    ax.barh(range(len(top_features)), top_importances, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str] = ["Real", "Fake"],
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_confusion_matrix(y_test, y_pred)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   model_name: str = "Model",
                   figsize: Tuple[int, int] = (8, 6),
                   save_path: Optional[Path] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_roc_curve(y_test, y_proba, "Random Forest")
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
           label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    plt.show()


def plot_temporal_propagation(df: pd.DataFrame,
                             timestamp_column: str = "timestamp",
                             label_column: str = "label",
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[Path] = None) -> None:
    """
    Plot temporal propagation patterns.
    
    Args:
        df: DataFrame with temporal data
        timestamp_column: Name of timestamp column
        label_column: Name of label column
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_temporal_propagation(df, "created_at", "is_fake")
    """
    logger.info("Plotting temporal propagation...")
    
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    df = df.sort_values(timestamp_column)
    
    df['hour'] = df[timestamp_column].dt.hour
    df['date'] = df[timestamp_column].dt.date
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Hourly distribution
    hourly_counts = df.groupby(['hour', label_column]).size().unstack(fill_value=0)
    hourly_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Posts by Hour of Day", fontsize=14, fontweight='bold')
    ax1.legend(["Real", "Fake"])
    ax1.tick_params(axis='x', rotation=45)
    
    # Daily trend
    daily_counts = df.groupby(['date', label_column]).size().unstack(fill_value=0)
    daily_counts.plot(ax=ax2, marker='o', color=['skyblue', 'lightcoral'])
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Daily Post Trends", fontsize=14, fontweight='bold')
    ax2.legend(["Real", "Fake"])
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved temporal plot to {save_path}")
    
    plt.show()


def plot_feature_correlations(feature_df: pd.DataFrame,
                             figsize: Tuple[int, int] = (12, 10),
                             save_path: Optional[Path] = None) -> None:
    """
    Plot feature correlation heatmap.
    
    Args:
        feature_df: DataFrame with features
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> plot_feature_correlations(features_df)
    """
    logger.info("Plotting feature correlations...")
    
    # Select numeric columns only
    numeric_df = feature_df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        logger.warning("No numeric columns found for correlation")
        return
    
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
    
    plt.show()


def create_interactive_network(G,
                              node_colors: Optional[Dict] = None,
                              node_sizes: Optional[Dict] = None,
                              output_path: Optional[Path] = None) -> None:
    """
    Create interactive network visualization with Plotly.
    
    Args:
        G: NetworkX graph
        node_colors: Dictionary mapping nodes to colors
        node_sizes: Dictionary mapping nodes to sizes
        output_path: Path to save HTML file
        
    Example:
        >>> create_interactive_network(G, node_colors, output_path=Path("network.html"))
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available. Cannot create interactive visualization.")
        return
    
    logger.info("Creating interactive network visualization...")
    
    # Get positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edge information
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        ))
    
    # Extract node information
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=[node_colors.get(node, 'lightblue') if node_colors else 'lightblue' 
                  for node in G.nodes()],
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                   layout=go.Layout(
                       title='Interactive Network Visualization',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive network to {output_path}")
    else:
        fig.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                          metric: str = "f1",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[Path] = None) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary mapping model names to metrics
        metric: Metric to compare
        figsize: Figure size
        save_path: Path to save the figure
        
    Example:
        >>> results = {"RF": {"f1": 0.85}, "XGB": {"f1": 0.87}}
        >>> plot_model_comparison(results, "f1")
    """
    logger.info(f"Plotting model comparison for {metric}...")
    
    model_names = list(results.keys())
    metric_values = [results[model].get(metric, 0) for model in model_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(model_names, metric_values, color='steelblue', alpha=0.7)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Model Comparison - {metric.upper()}", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.VIZ_CONFIG.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()

