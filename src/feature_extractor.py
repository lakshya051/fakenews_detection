"""
Feature Engineering Module for Misinformation Prediction.

This module extracts network, content, user, and temporal features
for the misinformation prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk

try:
    from transformers import BertTokenizer, BertModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers library not available. BERT features will be disabled.")

import config

logging.basicConfig(**config.LOG_CONFIG)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_NLTK_SENTIMENT = True
except:
    HAS_NLTK_SENTIMENT = False
    logger.warning("NLTK sentiment analyzer not available")


class FeatureExtractor:
    """Main class for extracting features from social media data."""
    
    def __init__(self, use_bert: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            use_bert: Whether to use BERT for text embeddings
        """
        self.use_bert = use_bert and HAS_TRANSFORMERS
        self.tfidf_vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        
        if self.use_bert:
            try:
                model_name = config.FEATURE_CONFIG.get("bert_model", "bert-base-uncased")
                self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
                self.bert_model = BertModel.from_pretrained(model_name)
                self.bert_model.eval()
                logger.info(f"Loaded BERT model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self.use_bert = False
        
        if HAS_NLTK_SENTIMENT:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
    
    def extract_all_features(self, 
                           df: pd.DataFrame,
                           network_features: Optional[pd.DataFrame] = None,
                           text_column: str = "text",
                           user_column: str = "user_id",
                           timestamp_column: str = "timestamp") -> pd.DataFrame:
        """
        Extract all features (network, content, user, temporal).
        
        Args:
            df: Input DataFrame
            network_features: Pre-computed network features DataFrame
            text_column: Name of text column
            user_column: Name of user ID column
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with all extracted features
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_all_features(df, network_features)
        """
        logger.info("Extracting all features...")
        
        feature_dfs = []
        
        # Network features
        if network_features is not None:
            feature_dfs.append(network_features)
        else:
            logger.warning("No network features provided")
        
        # Content features
        if text_column in df.columns:
            content_features = self.extract_content_features(df, text_column)
            feature_dfs.append(content_features)
        
        # User features
        if user_column in df.columns:
            user_features = self.extract_user_features(df, user_column)
            feature_dfs.append(user_features)
        
        # Temporal features
        if timestamp_column in df.columns:
            temporal_features = self.extract_temporal_features(df, timestamp_column)
            feature_dfs.append(temporal_features)
        
        # Combine all features
        if feature_dfs:
            result = pd.concat(feature_dfs, axis=1)
            logger.info(f"Extracted {len(result.columns)} features")
            return result
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()
    
    def extract_content_features(self, 
                                df: pd.DataFrame,
                                text_column: str = "text") -> pd.DataFrame:
        """
        Extract content-based features from text.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            DataFrame with content features
        """
        logger.info("Extracting content features...")
        
        features = {}
        
        for idx, row in df.iterrows():
            text = str(row.get(text_column, ""))
            
            # Text statistics
            features.setdefault("text_length", []).append(len(text))
            features.setdefault("word_count", []).append(len(text.split()))
            features.setdefault("char_count", []).append(len(text.replace(" ", "")))
            features.setdefault("sentence_count", []).append(len(text.split('.')))
            features.setdefault("exclamation_count", []).append(text.count('!'))
            features.setdefault("question_count", []).append(text.count('?'))
            features.setdefault("uppercase_ratio", []).append(
                sum(1 for c in text if c.isupper()) / len(text) if text else 0
            )
            
            # Sentiment features
            sentiment = self.extract_sentiment(text)
            features.setdefault("sentiment_polarity", []).append(sentiment["polarity"])
            features.setdefault("sentiment_subjectivity", []).append(sentiment["subjectivity"])
            
            if self.sentiment_analyzer:
                nltk_sentiment = self.sentiment_analyzer.polarity_scores(text)
                features.setdefault("sentiment_compound", []).append(nltk_sentiment["compound"])
                features.setdefault("sentiment_pos", []).append(nltk_sentiment["pos"])
                features.setdefault("sentiment_neg", []).append(nltk_sentiment["neg"])
                features.setdefault("sentiment_neu", []).append(nltk_sentiment["neu"])
            else:
                features.setdefault("sentiment_compound", []).append(0)
                features.setdefault("sentiment_pos", []).append(0)
                features.setdefault("sentiment_neg", []).append(0)
                features.setdefault("sentiment_neu", []).append(0)
            
            # Emotion features (simple keyword-based)
            emotion = self.extract_emotion_scores(text)
            features.setdefault("emotion_anger", []).append(emotion["anger"])
            features.setdefault("emotion_fear", []).append(emotion["fear"])
            features.setdefault("emotion_joy", []).append(emotion["joy"])
            features.setdefault("emotion_sadness", []).append(emotion["sadness"])
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
        except:
            return {"polarity": 0.0, "subjectivity": 0.0}
    
    def extract_emotion_scores(self, text: str) -> Dict[str, float]:
        """Extract emotion scores using keyword matching."""
        text_lower = text.lower()
        
        anger_words = ["angry", "rage", "furious", "hate", "outrage"]
        fear_words = ["afraid", "scared", "worried", "anxious", "panic"]
        joy_words = ["happy", "joy", "excited", "celebrate", "love"]
        sadness_words = ["sad", "depressed", "grief", "sorrow", "unhappy"]
        
        emotions = {
            "anger": sum(1 for word in anger_words if word in text_lower) / len(text.split()) if text else 0,
            "fear": sum(1 for word in fear_words if word in text_lower) / len(text.split()) if text else 0,
            "joy": sum(1 for word in joy_words if word in text_lower) / len(text.split()) if text else 0,
            "sadness": sum(1 for word in sadness_words if word in text_lower) / len(text.split()) if text else 0
        }
        
        return emotions
    
    def extract_tfidf_features(self, 
                              texts: List[str],
                              max_features: int = 5000) -> pd.DataFrame:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: List of text strings
            max_features: Maximum number of TF-IDF features
            
        Returns:
            DataFrame with TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )
    
    def extract_bert_embeddings(self, 
                               texts: List[str],
                               max_length: int = 128,
                               batch_size: int = 32) -> np.ndarray:
        """
        Extract BERT embeddings from texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of BERT embeddings
        """
        if not self.use_bert:
            logger.warning("BERT not available, returning zeros")
            return np.zeros((len(texts), 768))
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Get embeddings
                outputs = self.bert_model(**encoded)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def extract_user_features(self, 
                             df: pd.DataFrame,
                             user_column: str = "user_id") -> pd.DataFrame:
        """
        Extract user-based features.
        
        Args:
            df: Input DataFrame
            user_column: Name of user ID column
            
        Returns:
            DataFrame with user features
        """
        logger.info("Extracting user features...")
        
        features = {}
        
        # Account age (if account_created column exists)
        if "account_created" in df.columns:
            df["account_created"] = pd.to_datetime(df["account_created"], errors='coerce')
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                features["account_age_days"] = (
                    (df["timestamp"] - df["account_created"]).dt.total_seconds() / 86400
                ).fillna(0)
        
        # Follower/following ratio
        if "follower_count" in df.columns and "following_count" in df.columns:
            features["follower_following_ratio"] = (
                df["follower_count"] / (df["following_count"] + 1)
            ).fillna(0)
        
        # Verification status
        if "verified" in df.columns:
            features["is_verified"] = df["verified"].astype(int)
        else:
            features["is_verified"] = [0] * len(df)
        
        # Activity features (if multiple posts per user)
        user_stats = df.groupby(user_column).agg({
            "timestamp": "count" if "timestamp" in df.columns else lambda x: 0
        }).rename(columns={"timestamp": "user_post_count"})
        
        if user_column in df.columns:
            features["user_post_count"] = df[user_column].map(
                user_stats["user_post_count"]
            ).fillna(0)
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_temporal_features(self, 
                                 df: pd.DataFrame,
                                 timestamp_column: str = "timestamp") -> pd.DataFrame:
        """
        Extract temporal features from timestamps.
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Extracting temporal features...")
        
        if timestamp_column not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_column}' not found")
            return pd.DataFrame()
        
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        
        features = {
            "hour_of_day": df[timestamp_column].dt.hour,
            "day_of_week": df[timestamp_column].dt.dayofweek,
            "day_of_month": df[timestamp_column].dt.day,
            "month": df[timestamp_column].dt.month,
            "is_weekend": (df[timestamp_column].dt.dayofweek >= 5).astype(int),
            "is_business_hours": (
                (df[timestamp_column].dt.hour >= 9) & 
                (df[timestamp_column].dt.hour < 17)
            ).astype(int)
        }
        
        return pd.DataFrame(features, index=df.index)
    
    def extract_network_features_from_graph(self,
                                           G,
                                           node_mapping: Dict[str, int],
                                           centrality_df: Optional[pd.DataFrame] = None,
                                           communities: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Extract network features for nodes.
        
        Args:
            G: NetworkX graph
            node_mapping: Dictionary mapping node IDs to DataFrame indices
            centrality_df: Pre-computed centrality measures
            communities: Community membership dictionary
            
        Returns:
            DataFrame with network features
        """
        logger.info("Extracting network features from graph...")
        
        if centrality_df is None:
            # Import here to avoid circular dependency
            try:
                from . import network_builder
                centrality_df = network_builder.calculate_centrality_measures(G)
            except ImportError:
                import network_builder
                centrality_df = network_builder.calculate_centrality_measures(G)
        
        # Map nodes to DataFrame indices
        features = {}
        
        for node, idx in node_mapping.items():
            if node in centrality_df.index:
                for col in centrality_df.columns:
                    features.setdefault(col, []).append(centrality_df.loc[node, col])
            else:
                # Default values for nodes not in graph
                for col in centrality_df.columns:
                    features.setdefault(col, []).append(0)
        
        features_df = pd.DataFrame(features)
        
        # Add community membership
        if communities:
            features_df["community_id"] = [
                communities.get(node, -1) 
                for node, _ in sorted(node_mapping.items(), key=lambda x: x[1])
            ]
        
        return features_df


def extract_cascade_features(cascades: Dict[str, List[Dict]],
                            post_id_column: str = "post_id") -> pd.DataFrame:
    """
    Extract features from information cascades.
    
    Args:
        cascades: Dictionary of cascade information
        post_id_column: Name of post ID column in original DataFrame
        
    Returns:
        DataFrame with cascade features
    """
    logger.info("Extracting cascade features...")
    
    features = {}
    
    for post_id, cascade_info in cascades.items():
        features.setdefault("cascade_depth", []).append(cascade_info.get("depth", 0))
        features.setdefault("cascade_breadth", []).append(cascade_info.get("breadth", 0))
        features.setdefault("propagation_speed", []).append(cascade_info.get("propagation_speed", 0))
    
    return pd.DataFrame(features, index=list(cascades.keys()))

