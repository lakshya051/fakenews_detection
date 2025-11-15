# Quick Start Guide

Get started with the Misinformation Prediction project in minutes!

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: Some optional dependencies may show warnings but won't break the project:
# - python-igraph (optional, for advanced community detection)
# - lightgbm (optional, for LightGBM model)
# - torch, torch-geometric (optional, for GNN models)
# - transformers (optional, for BERT embeddings)
```

## ⚡ Quick Test

Run the example script to test everything:

```bash
python example_usage.py
```

This will:
1. Create sample data
2. Build a social network
3. Extract features
4. Train a model
5. Evaluate and visualize results

## 📓 Run Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open notebooks in order:
   - `notebooks/01_data_exploration.ipynb` - Explore your data
   - `notebooks/02_network_analysis.ipynb` - Build and analyze networks
   - `notebooks/03_feature_engineering.ipynb` - Extract features
   - `notebooks/04_modeling.ipynb` - Train models
   - `notebooks/05_visualization.ipynb` - Create visualizations

## 💻 Basic Usage

```python
from src import data_preprocessing, network_builder, feature_extractor, models

# 1. Load data
df = data_preprocessing.create_sample_dataset(n_samples=1000)

# 2. Build network
G = network_builder.build_interaction_graph(df, user_column="user_id")

# 3. Extract features
extractor = feature_extractor.FeatureExtractor(use_bert=False)
features = extractor.extract_all_features(df)

# 4. Train model
X = features.values
y = df['label'].values
model = models.TraditionalMLModel("random_forest")
model.train(X, y)

# 5. Evaluate
results = model.evaluate(X, y)
print(results)
```

## 📊 Using Your Own Data

```python
from pathlib import Path
from src import data_preprocessing

# Load your dataset
df = data_preprocessing.load_dataset(Path("data/raw/your_data.csv"))

# Or use Kaggle format
df = data_preprocessing.load_kaggle_dataset(Path("data/raw/kaggle_data.csv"))

# Ensure your data has:
# - 'text' column (news/article text)
# - 'label' column (0=real, 1=fake)
# Optional: 'user_id', 'timestamp', etc.
```

## 🎯 Next Steps

1. **Choose a dataset** - See [DATASETS.md](DATASETS.md) for recommendations
2. **Run notebooks** - Follow the pipeline step by step
3. **Customize** - Modify `config.py` for your needs
4. **Experiment** - Try different models and features

## ❓ Troubleshooting

**Import errors?**
- Make sure you're in the project root directory
- Install missing dependencies: `pip install <package-name>`

**Module not found?**
- Run: `python -c "from src import data_preprocessing; print('OK')"`
- Check that you're in the correct directory

**Dataset issues?**
- Use `create_sample_dataset()` to test first
- Check your data has required columns (text, label)

## 📚 More Information

- Full documentation: [README.md](README.md)
- Dataset guide: [DATASETS.md](DATASETS.md)
- Example usage: `example_usage.py`

Happy analyzing! 🎉

