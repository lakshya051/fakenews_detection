"""
Example Usage Script for Misinformation Prediction Project

This script demonstrates how to use the project modules to:
1. Load and preprocess data
2. Build social networks
3. Extract features
4. Train models
5. Evaluate and visualize results
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src import (
    data_preprocessing,
    network_builder,
    feature_extractor,
    models,
    visualization
)
import config


def main():
    """Main example workflow."""
    
    print("="*60)
    print("Misinformation Prediction - Example Usage")
    print("="*60)
    
    # Step 1: Load or create data
    print("\n[Step 1] Loading data...")
    df = data_preprocessing.create_sample_dataset(n_samples=1000)
    print(f"Loaded {len(df)} samples")
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    train_df, val_df, test_df = data_preprocessing.preprocess_dataset(
        df,
        text_column="text",
        label_column="label",
        timestamp_column="timestamp",
        save_processed=False
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 3: Build network
    print("\n[Step 3] Building social network...")
    G = network_builder.build_interaction_graph(
        df,
        user_column="user_id",
        directed=True
    )
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 4: Calculate network metrics
    print("\n[Step 4] Calculating network metrics...")
    stats = network_builder.calculate_network_statistics(G)
    print(f"Density: {stats['density']:.4f}")
    
    centrality_df = network_builder.calculate_centrality_measures(G)
    communities = network_builder.detect_communities(G)
    print(f"Communities detected: {len(set(communities.values()))}")
    
    # Step 5: Extract features
    print("\n[Step 5] Extracting features...")
    extractor = feature_extractor.FeatureExtractor(use_bert=False)
    
    train_features = extractor.extract_all_features(
        train_df,
        text_column="text",
        user_column="user_id",
        timestamp_column="timestamp"
    )
    
    test_features = extractor.extract_all_features(
        test_df,
        text_column="text",
        user_column="user_id",
        timestamp_column="timestamp"
    )
    
    print(f"Extracted {len(train_features.columns)} features")
    
    # Step 6: Train model
    print("\n[Step 6] Training model...")
    X_train = train_features.values
    y_train = train_df['label'].values
    X_test = test_features.values
    y_test = test_df['label'].values
    
    model = models.TraditionalMLModel("random_forest", n_estimators=100)
    model.train(X_train, y_train)
    
    # Step 7: Evaluate
    print("\n[Step 7] Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print("\nResults:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric:20s}: {value:.4f}")
    
    # Step 8: Visualize
    print("\n[Step 8] Creating visualizations...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    visualization.plot_confusion_matrix(y_test, y_pred, class_names=["Real", "Fake"])
    
    # ROC curve
    visualization.plot_roc_curve(y_test, y_proba, "Random Forest")
    
    # Feature importance
    feature_names = train_features.columns.tolist()
    visualization.plot_feature_importance(model.model, feature_names, top_n=10)
    
    print("\n" + "="*60)
    print("Example workflow completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

