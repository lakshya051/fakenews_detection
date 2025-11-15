"""
Configuration file for Misinformation Prediction Project.
Contains hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NETWORKS_DIR = DATA_DIR / "networks"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data paths
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Dataset configuration
DATASET_SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42

# FakeNewsNet dataset URLs (example)
FAKENEWSNET_URLS = {
    "gossipcop": "https://github.com/KaiDMML/FakeNewsNet",
    "politifact": "https://github.com/KaiDMML/FakeNewsNet"
}

# Text preprocessing
TEXT_CLEANING_CONFIG = {
    "remove_urls": True,
    "remove_mentions": False,  # Keep for network analysis
    "remove_hashtags": False,  # Keep for analysis
    "lowercase": True,
    "remove_special_chars": True,
    "min_text_length": 10
}

# Network configuration
NETWORK_CONFIG = {
    "directed": True,
    "weighted": True,
    "min_edge_weight": 1,
    "community_algorithm": "louvain"
}

# Feature engineering
FEATURE_CONFIG = {
    "use_bert": True,
    "bert_model": "bert-base-uncased",
    "max_sequence_length": 128,
    "tfidf_max_features": 5000,
    "use_sentiment": True,
    "use_emotion": True
}

# Model hyperparameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 5,
        "random_state": RANDOM_SEED
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_SEED
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "probability": True,
        "random_state": RANDOM_SEED
    },
    "gnn": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.5,
        "learning_rate": 0.01,
        "epochs": 100,
        "early_stopping_patience": 10,
        "batch_size": 32
    }
}

# Training configuration
TRAINING_CONFIG = {
    "cross_validation_folds": 5,
    "early_stopping": True,
    "save_best_model": True,
    "model_save_dir": PROJECT_ROOT / "models"
}

# Visualization
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "Set2"
}

# Logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
LOG_FILE = PROJECT_ROOT / "logs" / "project.log"

# Create necessary directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, NETWORKS_DIR, 
                  TRAINING_CONFIG["model_save_dir"], 
                  LOG_FILE.parent]:
    directory.mkdir(parents=True, exist_ok=True)

