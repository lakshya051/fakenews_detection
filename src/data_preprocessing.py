"""
Data Preprocessing Module for Misinformation Prediction.

This module handles data loading, cleaning, and preparation for the
misinformation prediction pipeline.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import requests
import zipfile
from sklearn.model_selection import train_test_split

import config

# Setup logging
logging.basicConfig(**config.LOG_CONFIG)
logger = logging.getLogger(__name__)


def download_fakenewsnet_dataset(dataset_type: str = "gossipcop", 
                                 output_dir: Optional[Path] = None) -> Path:
    """
    Download FakeNewsNet dataset.
    
    Note: This is a placeholder function. In practice, you would need to
    clone the FakeNewsNet repository or download from their official source.
    
    Args:
        dataset_type: Type of dataset ('gossipcop' or 'politifact')
        output_dir: Directory to save the dataset
        
    Returns:
        Path to downloaded dataset directory
        
    Example:
        >>> dataset_path = download_fakenewsnet_dataset("gossipcop")
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR / dataset_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading FakeNewsNet {dataset_type} dataset...")
    logger.warning("This is a placeholder. Please download the dataset manually from:")
    logger.warning(f"  {config.FAKENEWSNET_URLS.get(dataset_type, 'N/A')}")
    logger.info(f"Expected output directory: {output_dir}")
    
    return output_dir


def load_dataset(file_path: Path, 
                 file_format: str = "csv") -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        file_path: Path to the dataset file
        file_format: Format of the file ('csv', 'json', 'parquet')
        
    Returns:
        Loaded DataFrame
        
    Example:
        >>> df = load_dataset(Path("data/raw/news_data.csv"))
    """
    try:
        if file_format == "csv":
            df = pd.read_csv(file_path)
        elif file_format == "json":
            df = pd.read_json(file_path)
        elif file_format == "parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def clean_text(text: str, 
               config_dict: Optional[Dict[str, Any]] = None) -> str:
    """
    Clean text data by removing URLs, special characters, etc.
    
    Args:
        text: Input text to clean
        config_dict: Configuration dictionary for cleaning options
        
    Returns:
        Cleaned text
        
    Example:
        >>> clean_text("Check out https://example.com! #news")
        'check out  news'
    """
    if pd.isna(text) or text == "":
        return ""
    
    if config_dict is None:
        config_dict = config.TEXT_CLEANING_CONFIG
    
    text = str(text)
    
    # Remove URLs
    if config_dict.get("remove_urls", True):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (optional - keep for network analysis)
    if config_dict.get("remove_mentions", False):
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (optional - keep for analysis)
    if config_dict.get("remove_hashtags", False):
        text = re.sub(r'#\w+', '', text)
    
    # Lowercase
    if config_dict.get("lowercase", True):
        text = text.lower()
    
    # Remove special characters but keep spaces
    if config_dict.get("remove_special_chars", True):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_timestamp(df: pd.DataFrame, 
                     timestamp_column: str = "timestamp") -> pd.DataFrame:
    """
    Extract and convert timestamp column to datetime format.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        
    Returns:
        DataFrame with converted timestamp
        
    Example:
        >>> df = extract_timestamp(df, "created_at")
    """
    if timestamp_column not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_column}' not found")
        return df
    
    try:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        logger.info(f"Converted {timestamp_column} to datetime format")
    except Exception as e:
        logger.error(f"Error converting timestamp: {e}")
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = "drop",
                         columns: Optional[list] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        columns: Specific columns to process (None for all columns)
        
    Returns:
        DataFrame with handled missing values
        
    Example:
        >>> df = handle_missing_values(df, strategy="fill_mean")
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    missing_counts = df_processed[columns].isnull().sum()
    
    if missing_counts.sum() > 0:
        logger.info(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        
        for col in columns:
            if df_processed[col].isnull().sum() > 0:
                if strategy == "drop":
                    df_processed = df_processed.dropna(subset=[col])
                elif strategy == "fill_mean" and df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                elif strategy == "fill_median" and df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif strategy == "fill_mode":
                    df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 0, inplace=True)
                else:
                    df_processed[col].fillna("", inplace=True)
    
    logger.info(f"Handled missing values using strategy: {strategy}")
    return df_processed


def create_train_val_test_splits(df: pd.DataFrame,
                                 target_column: str = "label",
                                 train_ratio: float = 0.70,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_state: int = 42,
                                 stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits from the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column for stratification
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by target column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Example:
        >>> train, val, test = create_train_val_test_splits(df, "is_fake")
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    stratify_col = df[target_column] if stratify and target_column in df.columns else None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    stratify_col_temp = temp_df[target_column] if stratify and target_column in temp_df.columns else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=stratify_col_temp
    )
    
    logger.info(f"Created splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def preprocess_dataset(df: pd.DataFrame,
                      text_column: str = "text",
                      label_column: str = "label",
                      timestamp_column: Optional[str] = "timestamp",
                      save_processed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline for the dataset.
    
    Args:
        df: Raw input DataFrame
        text_column: Name of text content column
        label_column: Name of label column (0=real, 1=fake)
        timestamp_column: Name of timestamp column (optional)
        save_processed: Whether to save processed data to disk
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Example:
        >>> train, val, test = preprocess_dataset(df, "content", "is_fake")
    """
    logger.info("Starting dataset preprocessing pipeline...")
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = handle_missing_values(df_processed, strategy="drop")
    
    # Clean text column
    if text_column in df_processed.columns:
        logger.info("Cleaning text data...")
        df_processed[text_column] = df_processed[text_column].apply(clean_text)
        
        # Remove rows with very short text
        min_length = config.TEXT_CLEANING_CONFIG.get("min_text_length", 10)
        df_processed = df_processed[df_processed[text_column].str.len() >= min_length]
        logger.info(f"Removed rows with text shorter than {min_length} characters")
    
    # Extract timestamp
    if timestamp_column and timestamp_column in df_processed.columns:
        df_processed = extract_timestamp(df_processed, timestamp_column)
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(
        df_processed,
        target_column=label_column,
        train_ratio=config.DATASET_SPLITS["train"],
        val_ratio=config.DATASET_SPLITS["val"],
        test_ratio=config.DATASET_SPLITS["test"],
        random_state=config.RANDOM_SEED
    )
    
    # Save processed data
    if save_processed:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
        val_df.to_csv(config.VAL_DATA_PATH, index=False)
        test_df.to_csv(config.TEST_DATA_PATH, index=False)
        logger.info(f"Saved processed data to {config.PROCESSED_DATA_DIR}")
    
    logger.info("Preprocessing pipeline completed successfully!")
    
    return train_df, val_df, test_df


def load_fakenewsnet_dataset(dataset_type: str = "gossipcop",
                            label: str = "fake",
                            base_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load FakeNewsNet dataset from CSV files.
    
    The CSV files contain news articles with tweet IDs. To get full social network
    data, you need to use the FakeNewsNet collection scripts with Twitter API.
    
    Args:
        dataset_type: Type of dataset ('gossipcop' or 'politifact')
        label: Label type ('fake' or 'real')
        base_path: Base path to FakeNewsNet-master directory
        
    Returns:
        DataFrame with columns: id, news_url, title, tweet_ids, label
        
    Example:
        >>> df_fake = load_fakenewsnet_dataset("gossipcop", "fake")
        >>> df_real = load_fakenewsnet_dataset("gossipcop", "real")
    """
    if base_path is None:
        base_path = config.PROJECT_ROOT / "FakeNewsNet-master" / "dataset"
    else:
        base_path = Path(base_path) / "dataset"
    
    filename = f"{dataset_type}_{label}.csv"
    file_path = base_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"FakeNewsNet dataset file not found: {file_path}\n"
            f"Please ensure FakeNewsNet-master/dataset/{filename} exists."
        )
    
    logger.info(f"Loading FakeNewsNet dataset: {filename}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Add label column
    df['label'] = 1 if label == "fake" else 0
    
    # Add dataset source
    df['dataset_source'] = dataset_type
    
    # Parse tweet_ids (tab-separated string) into list
    if 'tweet_ids' in df.columns:
        df['tweet_ids_list'] = df['tweet_ids'].apply(
            lambda x: str(x).split('\t') if pd.notna(x) else []
        )
        df['num_tweets'] = df['tweet_ids_list'].apply(len)
    
    # Use title as text if text column doesn't exist
    if 'text' not in df.columns:
        df['text'] = df.get('title', '')
    
    logger.info(f"Loaded {len(df)} samples from {filename}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def load_kaggle_dataset(file_path: Path,
                       text_column: str = "text",
                       label_column: str = "label") -> pd.DataFrame:
    """
    Load dataset from Kaggle or similar format.
    
    Args:
        file_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Loaded and standardized DataFrame
        
    Example:
        >>> df = load_kaggle_dataset(Path("data/raw/kaggle_fake_news.csv"))
    """
    df = load_dataset(file_path, file_format="csv")
    
    # Standardize column names if needed
    column_mapping = {
        "article_text": "text",
        "content": "text",
        "news": "text",
        "is_fake": "label",
        "fake": "label",
        "target": "label"
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    logger.info(f"Loaded Kaggle dataset with {len(df)} rows")
    return df


def create_sample_dataset(n_samples: int = 1000,
                          output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the sample dataset
        
    Returns:
        Generated DataFrame
        
    Example:
        >>> sample_df = create_sample_dataset(1000)
    """
    np.random.seed(config.RANDOM_SEED)
    
    # Generate sample data
    fake_texts = [
        "Breaking: Shocking discovery that will change everything!",
        "You won't believe what happened next!",
        "Doctors hate this one trick!",
        "This secret they don't want you to know!",
        "Amazing cure discovered by accident!"
    ]
    
    real_texts = [
        "The weather forecast predicts rain for tomorrow.",
        "Local school board meeting scheduled for next week.",
        "New study published in scientific journal.",
        "City council approves budget for infrastructure.",
        "Annual community festival dates announced."
    ]
    
    data = []
    for i in range(n_samples):
        is_fake = np.random.choice([0, 1], p=[0.5, 0.5])
        text_pool = fake_texts if is_fake else real_texts
        text = np.random.choice(text_pool) + f" Sample text {i}."
        
        data.append({
            "id": f"post_{i}",
            "text": text,
            "label": is_fake,
            "user_id": f"user_{np.random.randint(0, 100)}",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            "retweet_count": np.random.randint(0, 1000),
            "like_count": np.random.randint(0, 5000),
            "follower_count": np.random.randint(0, 100000)
        })
    
    df = pd.DataFrame(data)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved sample dataset to {output_path}")
    
    return df

