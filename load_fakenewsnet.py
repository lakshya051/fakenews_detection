"""
Script to load and combine FakeNewsNet datasets for the misinformation prediction project.

This script loads all FakeNewsNet CSV files and combines them into a single dataset
ready for use with the project pipeline.
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent))
from src import data_preprocessing
import config


def load_all_fakenewsnet_data(base_path: Path = None) -> pd.DataFrame:
    """
    Load all FakeNewsNet datasets (gossipcop + politifact, fake + real).
    
    Args:
        base_path: Path to FakeNewsNet-master directory
        
    Returns:
        Combined DataFrame with all samples
    """
    if base_path is None:
        base_path = config.PROJECT_ROOT / "FakeNewsNet-master"
    
    datasets = []
    
    # Load all combinations
    for dataset_type in ["gossipcop", "politifact"]:
        for label in ["fake", "real"]:
            try:
                df = data_preprocessing.load_fakenewsnet_dataset(
                    dataset_type=dataset_type,
                    label=label,
                    base_path=base_path
                )
                datasets.append(df)
                print(f"[OK] Loaded {dataset_type} {label}: {len(df)} samples")
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                continue
    
    if not datasets:
        raise FileNotFoundError(
            "No FakeNewsNet datasets found. Please ensure CSV files exist in "
            f"{base_path}/dataset/"
        )
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Total samples loaded: {len(combined_df)}")
    print(f"Label distribution:")
    print(combined_df['label'].value_counts())
    print(f"\nDataset source distribution:")
    print(combined_df['dataset_source'].value_counts())
    print(f"{'='*60}\n")
    
    return combined_df


def prepare_fakenewsnet_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare FakeNewsNet data for network analysis.
    
    Since the CSV files only contain tweet IDs (not actual tweet content or user data),
    this function creates placeholder data for demonstration. In practice, you would
    need to use the FakeNewsNet collection scripts with Twitter API to get full data.
    
    Args:
        df: DataFrame from load_all_fakenewsnet_data()
        
    Returns:
        DataFrame with columns suitable for the project pipeline
    """
    # Create user_id from tweet_ids (simulated - in practice, get from Twitter API)
    # For now, we'll create synthetic user IDs based on tweet IDs
    def extract_user_id(row_id):
        try:
            parts = str(row_id).split('-')
            if len(parts) > 1:
                return f"user_{parts[1]}"
            else:
                return f"user_{row_id}"
        except:
            return f"user_{row_id}"
    
    df['user_id'] = df['id'].apply(extract_user_id)
    
    # Create post_id
    df['post_id'] = df['id']
    
    # Create timestamp (simulated - in practice, get from tweet data)
    import pandas as pd
    import numpy as np
    df['timestamp'] = [
        pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        for _ in range(len(df))
    ]
    
    # Add interaction data (simulated)
    import numpy as np
    df['interaction_type'] = np.random.choice(
        ['retweet', 'reply', 'mention'], 
        size=len(df)
    )
    # Sample target users with replacement to match length
    df['target_user_id'] = df['user_id'].sample(n=len(df), replace=True).values
    
    # Add user metadata (simulated - in practice, get from Twitter API)
    df['follower_count'] = np.random.randint(0, 100000, size=len(df))
    df['following_count'] = np.random.randint(0, 5000, size=len(df))
    df['verified'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    return df


if __name__ == "__main__":
    print("Loading FakeNewsNet datasets...")
    print("="*60)
    
    try:
        # Load all datasets
        df = load_all_fakenewsnet_data()
        
        # Prepare for analysis
        print("\nPreparing data for analysis...")
        df_prepared = prepare_fakenewsnet_for_analysis(df)
        
        # Save to processed data directory
        output_path = config.PROCESSED_DATA_DIR / "fakenewsnet_combined.csv"
        df_prepared.to_csv(output_path, index=False)
        
        print(f"\n[OK] Data saved to: {output_path}")
        print(f"[OK] Ready to use with the project pipeline!")
        print("\nNote: This uses simulated user/timestamp data.")
        print("For real social network analysis, use FakeNewsNet collection scripts")
        print("with Twitter API to get actual tweet and user data.")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nMake sure FakeNewsNet-master/dataset/ contains the CSV files:")
        print("  - gossipcop_fake.csv")
        print("  - gossipcop_real.csv")
        print("  - politifact_fake.csv")
        print("  - politifact_real.csv")

