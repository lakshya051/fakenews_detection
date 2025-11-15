"""
Modeling Module for Misinformation Prediction.

This module implements traditional ML models, Graph Neural Networks,
and hybrid models for misinformation detection.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("lightgbm not available. LightGBM model will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logging.warning("PyTorch/PyTorch Geometric not available. GNN models will be disabled.")

import config

logging.basicConfig(**config.LOG_CONFIG)
logger = logging.getLogger(__name__)


class TraditionalMLModel:
    """Wrapper class for traditional ML models."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize traditional ML model.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'logistic_regression', 'svm')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        model_config = config.MODEL_CONFIG.get(model_type, {})
        model_config.update(kwargs)
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**model_config)
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(**model_config)
        elif model_type == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("lightgbm is required for LightGBM model. Install it with: pip install lightgbm")
            self.model = lgb.LGBMClassifier(**model_config)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**model_config)
        elif model_type == "svm":
            self.model = SVC(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type} model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred = self.model.predict(X_train)
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            train_metrics.update({
                "val_accuracy": accuracy_score(y_val, val_pred),
                "val_f1": f1_score(y_val, val_pred)
            })
        
        logger.info(f"Training completed. Metrics: {train_metrics}")
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) > 1 else None
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)
        
        return metrics
    
    def save(self, filepath: Path) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class GCNModel(nn.Module):
    """Graph Convolutional Network for node classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 2,
                 dropout: float = 0.5):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, num_classes))
        else:
            self.convs.append(GCNConv(input_dim, num_classes))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return F.log_softmax(x, dim=1)


class GNNModel:
    """Wrapper class for Graph Neural Network models."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 2,
                 learning_rate: float = 0.01, device: str = "cpu"):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_classes: Number of output classes
            learning_rate: Learning rate
            device: Device to run on ('cpu' or 'cuda')
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch and PyTorch Geometric required for GNN models")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = GCNModel(input_dim, hidden_dim, num_layers, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.is_trained = False
        
        logger.info(f"Initialized GNN model on {self.device}")
    
    def train_model(self, data_list: List[Any], y_train: np.ndarray,
                   epochs: int = 100, early_stopping_patience: int = 10,
                   val_data_list: Optional[List[Any]] = None,
                   val_y: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train GNN model.
        
        Args:
            data_list: List of PyTorch Geometric Data objects
            y_train: Training labels
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            val_data_list: Validation data (optional)
            val_y: Validation labels (optional)
            
        Returns:
            Dictionary with training history
        """
        logger.info("Training GNN model...")
        
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Batch data
            batch = Batch.from_data_list(data_list).to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            
            total_loss = loss.item()
            history["train_loss"].append(total_loss)
            
            # Validation
            if val_data_list is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    val_batch = Batch.from_data_list(val_data_list).to(self.device)
                    val_out = self.model(val_batch.x, val_batch.edge_index, val_batch.batch)
                    val_y_tensor = torch.LongTensor(val_y).to(self.device)
                    val_loss = F.nll_loss(val_out, val_y_tensor).item()
                    history["val_loss"].append(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        self.is_trained = True
        logger.info("GNN training completed")
        return history
    
    def predict(self, data_list: List[Any]) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list(data_list).to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1).cpu().numpy()
        return pred
    
    def predict_proba(self, data_list: List[Any]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list(data_list).to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            proba = torch.exp(out).cpu().numpy()
        return proba


class HybridModel:
    """Hybrid model combining GNN and BERT embeddings."""
    
    def __init__(self, gnn_input_dim: int, bert_dim: int = 768,
                 hidden_dim: int = 128, num_classes: int = 2,
                 learning_rate: float = 0.001, device: str = "cpu"):
        """
        Initialize hybrid model.
        
        Args:
            gnn_input_dim: GNN input feature dimension
            bert_dim: BERT embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            learning_rate: Learning rate
            device: Device to run on
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch required for hybrid model")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # GNN component
        self.gnn = GCNModel(gnn_input_dim, hidden_dim, 2, hidden_dim).to(self.device)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.fusion.parameters()),
            lr=learning_rate
        )
        self.is_trained = False
        
        logger.info("Initialized hybrid model")
    
    def forward(self, graph_data: Any, bert_embeddings: Any) -> Any:
        """Forward pass."""
        # GNN output
        gnn_out = self.gnn(graph_data.x, graph_data.edge_index)
        gnn_pooled = global_mean_pool(gnn_out, graph_data.batch)
        
        # Concatenate with BERT
        combined = torch.cat([gnn_pooled, bert_embeddings], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        return F.log_softmax(output, dim=1)
    
    def train_model(self, graph_data_list: List[Any], bert_embeddings: np.ndarray,
                   y_train: np.ndarray, epochs: int = 100,
                   val_graph_data: Optional[List[Any]] = None,
                   val_bert_embeddings: Optional[np.ndarray] = None,
                   val_y: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train hybrid model."""
        logger.info("Training hybrid model...")
        
        history = {"train_loss": [], "val_loss": []}
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        bert_embeddings_tensor = torch.FloatTensor(bert_embeddings).to(self.device)
        
        for epoch in range(epochs):
            self.gnn.train()
            self.fusion.train()
            
            total_loss = 0
            batch = Batch.from_data_list(graph_data_list).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.forward(batch, bert_embeddings_tensor)
            loss = F.nll_loss(output, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            
            total_loss = loss.item()
            history["train_loss"].append(total_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        self.is_trained = True
        return history


def train_with_cross_validation(model: TraditionalMLModel,
                                X: np.ndarray, y: np.ndarray,
                                cv_folds: int = 5) -> Dict[str, float]:
    """
    Train model with cross-validation.
    
    Args:
        model: Model instance
        X: Features
        y: Labels
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with CV metrics
    """
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_SEED)
    
    scores = cross_val_score(model.model, X, y, cv=cv, scoring='f1')
    
    return {
        "cv_mean_f1": scores.mean(),
        "cv_std_f1": scores.std(),
        "cv_scores": scores.tolist()
    }


def evaluate_model(model: Union[TraditionalMLModel, GNNModel],
                   X_test: np.ndarray, y_test: np.ndarray,
                   X_test_graph: Optional[List] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_test_graph: Test graph data (for GNN)
        
    Returns:
        Dictionary with evaluation metrics and reports
    """
    logger.info("Evaluating model...")
    
    if isinstance(model, GNNModel):
        y_pred = model.predict(X_test_graph)
        y_proba = model.predict_proba(X_test_graph)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    
    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics


def early_detection_analysis(model: Union[TraditionalMLModel, GNNModel],
                            X_test: np.ndarray, y_test: np.ndarray,
                            time_hours: List[int],
                            timestamps: Optional[np.ndarray] = None) -> Dict[int, Dict[str, float]]:
    """
    Analyze model performance for early detection (within first N hours).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        time_hours: List of time thresholds in hours
        timestamps: Timestamps for test data
        
    Returns:
        Dictionary mapping hours to performance metrics
    """
    logger.info("Performing early detection analysis...")
    
    results = {}
    
    if timestamps is None:
        logger.warning("No timestamps provided, using all data")
        for hours in time_hours:
            results[hours] = evaluate_model(model, X_test, y_test)
        return results
    
    timestamps = pd.to_datetime(timestamps)
    time_diffs = (timestamps - timestamps.min()).dt.total_seconds() / 3600
    
    for hours in time_hours:
        mask = time_diffs <= hours
        if mask.sum() > 0:
            X_subset = X_test[mask]
            y_subset = y_test[mask]
            results[hours] = evaluate_model(model, X_subset, y_subset)
        else:
            results[hours] = {"error": "No data within time threshold"}
    
    return results

