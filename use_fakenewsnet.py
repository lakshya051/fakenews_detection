"""
Complete workflow to use FakeNewsNet dataset WITHOUT Twitter API.

This script loads the FakeNewsNet CSV files and prepares them for the
complete misinformation prediction pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))
from src import data_preprocessing, network_builder, feature_extractor, models, visualization
import config


def load_and_prepare_fakenewsnet():
    """Load and prepare FakeNewsNet data for analysis."""
    
    print("="*70)
    print("FakeNewsNet Dataset - Complete Workflow (No API Required)")
    print("="*70)
    
    # Step 1: Load all CSV files
    print("\n[Step 1] Loading FakeNewsNet CSV files...")
    datasets = []
    
    for dataset_type in ["gossipcop", "politifact"]:
        for label in ["fake", "real"]:
            try:
                df = data_preprocessing.load_fakenewsnet_dataset(
                    dataset_type=dataset_type,
                    label=label
                )
                datasets.append(df)
                print(f"  [OK] {dataset_type} {label}: {len(df)} samples")
            except Exception as e:
                print(f"  [ERROR] Error loading {dataset_type} {label}: {e}")
    
    if not datasets:
        raise FileNotFoundError("No datasets loaded!")
    
    # Combine all datasets
    df = pd.concat(datasets, ignore_index=True)
    print(f"\n  Total samples: {len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Step 2: Prepare data for network analysis
    print("\n[Step 2] Preparing data for network analysis...")
    
    # Create user IDs from news IDs (simulated users)
    def create_user_id(news_id):
        parts = str(news_id).split('-')
        if len(parts) > 1:
            return f"user_{parts[1][:8]}"
        return f"user_{hash(news_id) % 10000}"
    
    df['user_id'] = df['id'].apply(create_user_id)
    df['post_id'] = df['id']
    
    # Create timestamps (distributed over past year)
    np.random.seed(42)
    days_ago = np.random.randint(0, 365, size=len(df))
    df['timestamp'] = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit='D')
    
    # Create interaction network (simulated from tweet_ids)
    print("  Creating interaction network from tweet IDs...")
    
    # Extract tweet IDs and create user interactions
    interactions = []
    for idx, row in df.iterrows():
        user = row['user_id']
        tweet_ids = row.get('tweet_ids_list', [])
        
        # Create interactions based on tweet IDs
        # Simulate that users interact with each other through tweets
        if tweet_ids and len(tweet_ids) > 0:
            # Create target users based on tweet IDs (simulated)
            for i, tweet_id in enumerate(tweet_ids[:5]):  # Limit to first 5 tweets
                target_user = f"user_{hash(tweet_id) % 10000}"
                interactions.append({
                    'user_id': user,
                    'target_user_id': target_user,
                    'interaction_type': np.random.choice(['retweet', 'reply', 'mention']),
                    'timestamp': row['timestamp'] + timedelta(minutes=i*10),
                    'post_id': row['post_id']
                })
    
    # Create interaction dataframe
    if interactions:
        interaction_df = pd.DataFrame(interactions)
        print(f"  Created {len(interaction_df)} simulated interactions")
    else:
        # Fallback: create random interactions
        interaction_df = pd.DataFrame({
            'user_id': df['user_id'].sample(n=min(5000, len(df)*3), replace=True).values,
            'target_user_id': df['user_id'].sample(n=min(5000, len(df)*3), replace=True).values,
            'interaction_type': np.random.choice(['retweet', 'reply', 'mention'], size=min(5000, len(df)*3)),
            'timestamp': df['timestamp'].sample(n=min(5000, len(df)*3), replace=True).values,
            'post_id': df['post_id'].sample(n=min(5000, len(df)*3), replace=True).values
        })
        print(f"  Created {len(interaction_df)} random interactions")
    
    # Add user metadata (simulated)
    # Get all unique user IDs from both df and interaction_df
    all_user_ids = set(df['user_id'].unique())
    if len(interaction_df) > 0:
        all_user_ids.update(interaction_df['user_id'].unique())
        all_user_ids.update(interaction_df['target_user_id'].unique())
    
    unique_user_list = list(all_user_ids)
    num_users = len(unique_user_list)
    
    user_metadata = pd.DataFrame({
        'user_id': unique_user_list,
        'follower_count': np.random.randint(10, 100000, size=num_users),
        'following_count': np.random.randint(10, 5000, size=num_users),
        'verified': np.random.choice([0, 1], size=num_users, p=[0.95, 0.05]),
        'account_created': [pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 3650)) for _ in range(num_users)]
    })
    
    # Merge user metadata
    df = df.merge(user_metadata, on='user_id', how='left')
    df['follower_count'] = df['follower_count'].fillna(100)
    df['following_count'] = df['following_count'].fillna(50)
    df['verified'] = df['verified'].fillna(0)
    
    print(f"  Added metadata for {len(user_metadata)} unique users")
    
    # Step 3: Preprocess and split data
    print("\n[Step 3] Preprocessing and splitting data...")
    train_df, val_df, test_df = data_preprocessing.preprocess_dataset(
        df,
        text_column="title",  # Use title as text
        label_column="label",
        timestamp_column="timestamp",
        save_processed=True
    )
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 4: Build network
    print("\n[Step 4] Building social network...")
    G = network_builder.build_interaction_graph(
        interaction_df,
        user_column="user_id",
        target_column="target_user_id",
        timestamp_column="timestamp",
        directed=True
    )
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 5: Calculate network features
    print("\n[Step 5] Calculating network metrics...")
    stats = network_builder.calculate_network_statistics(G)
    print(f"  Density: {stats['density']:.4f}")
    
    centrality_df = network_builder.calculate_centrality_measures(G)
    communities = network_builder.detect_communities(G)
    print(f"  Communities: {len(set(communities.values()))}")
    
    # Step 6: Extract features
    print("\n[Step 6] Extracting features...")
    extractor = feature_extractor.FeatureExtractor(use_bert=False)
    
    train_features = extractor.extract_all_features(
        train_df,
        text_column="title",
        user_column="user_id",
        timestamp_column="timestamp"
    )
    
    val_features = extractor.extract_all_features(
        val_df,
        text_column="title",
        user_column="user_id",
        timestamp_column="timestamp"
    )
    
    test_features = extractor.extract_all_features(
        test_df,
        text_column="title",
        user_column="user_id",
        timestamp_column="timestamp"
    )
    
    print(f"  Extracted {len(train_features.columns)} features")
    
    # Step 7: Train model
    print("\n[Step 7] Training model...")
    X_train = train_features.values
    y_train = train_df['label'].values
    X_val = val_features.values
    y_val = val_df['label'].values
    X_test = test_features.values
    y_test = test_df['label'].values
    
    model = models.TraditionalMLModel("random_forest", n_estimators=100, max_depth=20)
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 8: Evaluate
    print("\n[Step 8] Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print("\n  Results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"    {metric:20s}: {value:.4f}")
    
    # Step 9: Visualize
    print("\n[Step 9] Creating visualizations...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Save visualizations
    viz_dir = Path("results")
    viz_dir.mkdir(exist_ok=True)
    
    visualization.plot_confusion_matrix(
        y_test, y_pred, 
        class_names=["Real", "Fake"],
        save_path=viz_dir / "confusion_matrix.png"
    )
    
    visualization.plot_roc_curve(
        y_test, y_proba, "Random Forest",
        save_path=viz_dir / "roc_curve.png"
    )
    
    feature_names = train_features.columns.tolist()
    visualization.plot_feature_importance(
        model.model, feature_names, top_n=15,
        save_path=viz_dir / "feature_importance.png"
    )
    
    print(f"  Visualizations saved to {viz_dir}/")
    
    # Step 10: Save everything
    print("\n[Step 10] Saving results...")
    
    # Save network
    network_builder.export_network(
        G, 
        config.NETWORKS_DIR / "fakenewsnet_network.graphml",
        format="graphml"
    )
    
    # Save model
    model_path = config.TRAINING_CONFIG["model_save_dir"] / "fakenewsnet_model.pkl"
    model.save(model_path)
    
    # Save results summary
    results_summary = {
        'dataset_size': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'network_nodes': G.number_of_nodes(),
        'network_edges': G.number_of_edges(),
        'num_features': len(train_features.columns),
        'model_performance': {k: v for k, v in results.items() if isinstance(v, (int, float))}
    }
    
    import json
    with open(viz_dir / "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  Model saved to: {model_path}")
    print(f"  Network saved to: {config.NETWORKS_DIR / 'fakenewsnet_network.graphml'}")
    print(f"  Results summary: {viz_dir / 'results_summary.json'}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Complete workflow finished successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check results/ folder for visualizations")
    print("  2. Open notebooks to explore the data further")
    print("  3. Try different models in notebook 04_modeling.ipynb")
    
    return df, G, model, results


if __name__ == "__main__":
    try:
        df, G, model, results = load_and_prepare_fakenewsnet()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

