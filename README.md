# Social Network Analysis for Misinformation News Prediction

A comprehensive Python project for predicting misinformation in social networks using network analysis, machine learning, and deep learning techniques.

## 📋 Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Modules](#modules)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Project Description

This project implements a complete pipeline for detecting misinformation in social media using:

- **Social Network Analysis**: Analyzing user interactions, information cascades, and network structure
- **Feature Engineering**: Extracting network, content, user, and temporal features
- **Machine Learning**: Traditional ML models (Random Forest, XGBoost, SVM, Logistic Regression)
- **Deep Learning**: Graph Neural Networks (GCN) and hybrid models combining GNN with BERT
- **Visualization**: Comprehensive visualizations of networks, features, and model performance

## ✨ Features

### Data Preprocessing
- Text cleaning and normalization
- Timestamp extraction and conversion
- Train/validation/test splits (70/15/15)
- Missing value handling
- Sample dataset generation for testing

### Network Construction
- Build directed graphs from user interactions
- Calculate network statistics (density, clustering, etc.)
- Compute centrality measures (degree, betweenness, closeness, eigenvector)
- Community detection using Louvain algorithm
- Information cascade identification
- Network export in multiple formats (GraphML, GML, JSON)

### Feature Engineering
- **Network Features**: Centrality scores, community membership, k-core, cascade metrics
- **Content Features**: Sentiment analysis, emotion scores, text statistics, TF-IDF, BERT embeddings
- **User Features**: Account age, follower/following ratio, verification status
- **Temporal Features**: Hour of day, day of week, posting frequency

### Models
- **Traditional ML**: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM
- **Graph Neural Networks**: GCN (Graph Convolutional Network) using PyTorch Geometric
- **Hybrid Models**: Combining GNN with BERT text embeddings
- Cross-validation support
- Early detection analysis
- Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)

### Visualization
- Network graphs with color-coded nodes
- Information diffusion cascades
- Feature importance plots
- Confusion matrices
- ROC curves
- Temporal propagation patterns
- Interactive network visualizations (Plotly)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd misinformation-prediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For PyTorch Geometric, you may need to install PyTorch first:

```bash
# For CPU
pip install torch torchvision torchaudio

# For GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch Geometric
pip install torch-geometric
```

### Step 3: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

## 📁 Project Structure

```
misinformation-prediction/
├── data/
│   ├── raw/              # Raw dataset files
│   ├── processed/        # Processed datasets (train/val/test splits)
│   └── networks/         # Exported network files
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data exploration and statistics
│   ├── 02_network_analysis.ipynb      # Network construction and analysis
│   ├── 03_feature_engineering.ipynb   # Feature extraction and analysis
│   ├── 04_modeling.ipynb              # Model training and evaluation
│   └── 05_visualization.ipynb         # Comprehensive visualizations
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── network_builder.py        # Network construction and analysis
│   ├── feature_extractor.py      # Feature engineering
│   ├── models.py                 # ML and DL models
│   └── visualization.py          # Visualization functions
├── tests/                  # Unit tests
├── config.py              # Configuration file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 💻 Usage

### Basic Workflow

#### 1. Data Preprocessing

```python
from src import data_preprocessing
from pathlib import Path

# Load dataset
df = data_preprocessing.load_dataset(Path("data/raw/news_data.csv"))

# Or create sample dataset for testing
df = data_preprocessing.create_sample_dataset(n_samples=1000)

# Preprocess and create splits
train_df, val_df, test_df = data_preprocessing.preprocess_dataset(
    df,
    text_column="text",
    label_column="label",
    timestamp_column="timestamp"
)
```

#### 2. Network Construction

```python
from src import network_builder

# Build interaction graph
G = network_builder.build_interaction_graph(
    df,
    user_column="user_id",
    interaction_column="interaction_type",
    directed=True
)

# Calculate network statistics
stats = network_builder.calculate_network_statistics(G)

# Calculate centrality measures
centrality_df = network_builder.calculate_centrality_measures(G)

# Detect communities
communities = network_builder.detect_communities(G, algorithm="louvain")
```

#### 3. Feature Engineering

```python
from src import feature_extractor

# Initialize extractor
extractor = feature_extractor.FeatureExtractor(use_bert=True)

# Extract all features
features = extractor.extract_all_features(
    df,
    network_features=centrality_df,
    text_column="text",
    user_column="user_id",
    timestamp_column="timestamp"
)
```

#### 4. Model Training

```python
from src import models
import numpy as np

# Prepare data
X_train = train_features.values
y_train = train_df['label'].values
X_test = test_features.values
y_test = test_df['label'].values

# Train Random Forest
rf_model = models.TraditionalMLModel("random_forest")
rf_model.train(X_train, y_train)
results = rf_model.evaluate(X_test, y_test)
print(results)

# Train XGBoost
xgb_model = models.TraditionalMLModel("xgboost")
xgb_model.train(X_train, y_train)
results = xgb_model.evaluate(X_test, y_test)
print(results)
```

#### 5. Visualization

```python
from src import visualization

# Plot network
visualization.plot_network_graph(
    G,
    node_colors=node_colors,
    title="Social Network"
)

# Plot confusion matrix
visualization.plot_confusion_matrix(y_test, y_pred)

# Plot ROC curve
visualization.plot_roc_curve(y_test, y_proba, "Random Forest")
```

### Running Jupyter Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory and open the notebooks in order:
   - `01_data_exploration.ipynb`
   - `02_network_analysis.ipynb`
   - `03_feature_engineering.ipynb`
   - `04_modeling.ipynb`
   - `05_visualization.ipynb`

## 📊 Dataset

### Recommended Datasets

This project works with multiple datasets. See **[DATASETS.md](DATASETS.md)** for a comprehensive guide.

**Quick Recommendations:**
- **FakeNewsNet** ⭐ (Best for social network analysis)
- **NELA-GT-2020** (Best for large-scale training)
- **CoAID** (Best for health misinformation)
- **Kaggle Fake News** (Best for quick start)

### Dataset Format

Expected columns:
- `text`: News article or post text
- `label`: Binary label (0=real, 1=fake)
- `user_id`: User identifier (optional but recommended)
- `timestamp`: Post timestamp (optional but recommended)
- `post_id`: Unique post identifier (optional)
- Additional columns: `follower_count`, `following_count`, `verified`, etc.

### Quick Start Options

**Option 1: Use Sample Data (Testing)**
```python
from src import data_preprocessing
df = data_preprocessing.create_sample_dataset(n_samples=1000)
```

**Option 2: Load Your Own Dataset**
```python
from src import data_preprocessing
from pathlib import Path

# Load from CSV
df = data_preprocessing.load_dataset(Path("data/raw/your_data.csv"))

# Or load Kaggle format
df = data_preprocessing.load_kaggle_dataset(Path("data/raw/kaggle_data.csv"))
```

**Option 3: Use FakeNewsNet**
1. Clone: `git clone https://github.com/KaiDMML/FakeNewsNet.git`
2. Follow their setup instructions
3. Load the data using `load_dataset()`

For detailed dataset information, see **[DATASETS.md](DATASETS.md)**.

## 🔧 Modules

### `data_preprocessing.py`
- `load_dataset()`: Load data from various formats
- `clean_text()`: Clean and normalize text
- `preprocess_dataset()`: Complete preprocessing pipeline
- `create_train_val_test_splits()`: Create data splits
- `create_sample_dataset()`: Generate sample data for testing

### `network_builder.py`
- `build_interaction_graph()`: Build network from interactions
- `calculate_network_statistics()`: Compute network metrics
- `calculate_centrality_measures()`: Calculate centrality scores
- `detect_communities()`: Community detection
- `identify_information_cascades()`: Find information cascades
- `export_network()`: Export network to file

### `feature_extractor.py`
- `FeatureExtractor`: Main feature extraction class
- `extract_content_features()`: Text-based features
- `extract_network_features()`: Network-based features
- `extract_user_features()`: User-based features
- `extract_temporal_features()`: Time-based features
- `extract_bert_embeddings()`: BERT text embeddings

### `models.py`
- `TraditionalMLModel`: Wrapper for traditional ML models
- `GNNModel`: Graph Neural Network implementation
- `HybridModel`: GNN + BERT hybrid model
- `train_with_cross_validation()`: Cross-validation training
- `evaluate_model()`: Comprehensive model evaluation
- `early_detection_analysis()`: Early detection performance

### `visualization.py`
- `plot_network_graph()`: Network visualization
- `plot_information_cascade()`: Cascade visualization
- `plot_feature_importance()`: Feature importance plots
- `plot_confusion_matrix()`: Confusion matrix
- `plot_roc_curve()`: ROC curve
- `plot_temporal_propagation()`: Temporal patterns
- `create_interactive_network()`: Interactive Plotly visualization

## 📈 Results

### Expected Performance Metrics

Typical performance on FakeNewsNet dataset:

- **Random Forest**: F1 ~0.85, Accuracy ~0.87
- **XGBoost**: F1 ~0.87, Accuracy ~0.89
- **GNN**: F1 ~0.88, Accuracy ~0.90
- **Hybrid (GNN + BERT)**: F1 ~0.90, Accuracy ~0.92

*Note: Actual results depend on dataset quality and hyperparameters.*

## 🔮 Future Improvements

1. **Additional Datasets**: Support for more misinformation datasets
2. **Advanced GNN Architectures**: GAT, GraphSAGE, Transformer-based GNNs
3. **Real-time Prediction**: Stream processing for live social media feeds
4. **Explainability**: SHAP values, attention visualization
5. **Web Application**: Streamlit dashboard for interactive analysis
6. **Multi-modal Features**: Image and video analysis
7. **Temporal GNNs**: Time-aware graph neural networks
8. **Active Learning**: Efficient annotation strategies
9. **Deployment**: Model serving with FastAPI or Flask
10. **Monitoring**: Model performance tracking and drift detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- FakeNewsNet dataset creators
- NetworkX and PyTorch Geometric communities
- All open-source libraries used in this project

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research project. Results may vary based on dataset and hyperparameters. Always validate models on your specific use case.

