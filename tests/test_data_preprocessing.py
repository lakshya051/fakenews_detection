"""
Unit tests for data_preprocessing module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src import data_preprocessing


def test_clean_text():
    """Test text cleaning function."""
    text = "Check out https://example.com! #news @user"
    cleaned = data_preprocessing.clean_text(text)
    assert isinstance(cleaned, str)
    assert "http" not in cleaned.lower()


def test_create_sample_dataset():
    """Test sample dataset creation."""
    df = data_preprocessing.create_sample_dataset(n_samples=100)
    assert len(df) == 100
    assert "text" in df.columns
    assert "label" in df.columns
    assert "user_id" in df.columns


def test_handle_missing_values():
    """Test missing value handling."""
    df = pd.DataFrame({
        "col1": [1, 2, None, 4],
        "col2": ["a", "b", "c", None]
    })
    df_processed = data_preprocessing.handle_missing_values(df, strategy="drop")
    assert df_processed.isnull().sum().sum() == 0


def test_create_train_val_test_splits():
    """Test data splitting."""
    df = data_preprocessing.create_sample_dataset(n_samples=1000)
    train, val, test = data_preprocessing.create_train_val_test_splits(df)
    
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(df)


if __name__ == "__main__":
    pytest.main([__file__])

