# Dataset Guide for Misinformation Prediction

This guide helps you choose the best dataset for your misinformation prediction project.

## 🏆 Recommended Datasets

### 1. **FakeNewsNet** ⭐ (Best for Social Network Analysis)

**Why it's great for this project:**
- ✅ Includes social network data (retweets, replies, user interactions)
- ✅ Has both news articles AND social media posts
- ✅ Two domains: GossipCop (entertainment) and PolitiFact (political)
- ✅ Well-documented and widely used in research
- ✅ Includes user metadata (followers, verification status, etc.)

**Dataset Details:**
- **GossipCop**: ~22,000 news articles with social engagement
- **PolitiFact**: ~1,000 news articles with social engagement
- **Format**: News articles + Twitter engagement data
- **Labels**: Real/Fake (verified by fact-checkers)

**How to Get It:**
```bash
# Clone the repository
git clone https://github.com/KaiDMML/FakeNewsNet.git
cd FakeNewsNet

# Follow their setup instructions
# You'll need Twitter API credentials for full dataset
# Or use their pre-processed data if available
```

**Pros:**
- Perfect for social network analysis (has retweet/reply networks)
- Includes temporal data (timestamps)
- User metadata available
- Active research community

**Cons:**
- Requires Twitter API access for full dataset
- Some data may be outdated (Twitter API changes)
- Setup can be complex

---

### 2. **NELA-GT-2020** ⭐ (Best for Large-Scale Analysis)

**Why it's great:**
- ✅ **HUGE dataset**: ~1.8 million articles from 519 sources
- ✅ Source-level ground truth labels
- ✅ Includes embedded tweets
- ✅ Covers entire year 2020
- ✅ Multiple fact-checking sources

**Dataset Details:**
- **Size**: ~1.8M articles
- **Sources**: 519 news outlets
- **Labels**: Source-level reliability ratings
- **Format**: News articles + metadata

**How to Get It:**
- Available on GitHub: https://github.com/emilykchen/NELA-GT-2020
- Direct download links provided

**Pros:**
- Massive scale for robust training
- Multiple fact-checking sources
- Good for generalizability studies
- Well-maintained

**Cons:**
- Source-level labels (not article-level)
- Less social network data than FakeNewsNet
- Large file size requires significant storage

---

### 3. **CoAID (COVID-19 Misinformation)** ⭐ (Best for Health Misinformation)

**Why it's great:**
- ✅ Focus on health misinformation (highly relevant)
- ✅ Includes social media posts
- ✅ ~296,000 user engagements
- ✅ Timely and important topic

**Dataset Details:**
- **Size**: 4,251 news articles + 926 social posts
- **Engagements**: ~296,000 user interactions
- **Domain**: COVID-19 healthcare misinformation
- **Labels**: Real/Fake

**How to Get It:**
- GitHub: https://github.com/cuilimeng/CoAID
- Direct download available

**Pros:**
- Health misinformation focus (critical area)
- Includes social engagement data
- Recent and relevant
- Good for domain-specific studies

**Cons:**
- Smaller than other datasets
- Domain-specific (may not generalize)
- COVID-19 focused (time-bound)

---

### 4. **Fake News Classification Dataset** (Best for Quick Start)

**Why it's great:**
- ✅ Easy to use
- ✅ Multiple sources combined
- ✅ Good size: 72,134 articles
- ✅ Ready-to-use format

**Dataset Details:**
- **Size**: 72,134 articles
- **Sources**: Kaggle, McIntire, Reuters, BuzzFeed
- **Format**: Clean CSV/JSON
- **Labels**: Factual/Misinformation

**How to Get It:**
- Available on Kaggle
- Multiple versions available

**Pros:**
- Easy to download and use
- Clean format
- Good for quick prototyping
- Multiple sources

**Cons:**
- Less social network data
- May need preprocessing
- Less metadata than specialized datasets

---

## 📊 Comparison Table

| Dataset | Size | Social Network Data | Ease of Use | Best For |
|---------|------|-------------------|-------------|----------|
| **FakeNewsNet** | Medium | ⭐⭐⭐⭐⭐ | Medium | Social network analysis |
| **NELA-GT-2020** | Very Large | ⭐⭐⭐ | Easy | Large-scale training |
| **CoAID** | Medium | ⭐⭐⭐⭐ | Easy | Health misinformation |
| **Fake News Classification** | Large | ⭐⭐ | Very Easy | Quick prototyping |

---

## 🎯 My Recommendation

### **For Your Project: Start with FakeNewsNet**

**Reasons:**
1. **Perfect for Social Network Analysis**: Has retweet/reply networks, user interactions
2. **Rich Metadata**: User profiles, timestamps, engagement metrics
3. **Research Standard**: Widely used, good for comparisons
4. **Matches Your Requirements**: Has everything your project needs

### **Alternative: Use Multiple Datasets**

Consider using:
- **FakeNewsNet** for network analysis
- **NELA-GT-2020** for large-scale training
- **CoAID** for domain-specific analysis

---

## 🚀 Quick Start Guide

### Option 1: Use Sample Data (Testing)

The project includes a function to generate sample data:

```python
from src import data_preprocessing

# Generate sample dataset
df = data_preprocessing.create_sample_dataset(n_samples=1000)
```

### Option 2: Use FakeNewsNet

1. **Clone FakeNewsNet repository:**
```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git
```

2. **Set up Twitter API** (if needed for full dataset)

3. **Load data:**
```python
from src import data_preprocessing
from pathlib import Path

# Load FakeNewsNet data
df = data_preprocessing.load_dataset(
    Path("path/to/fakenewsnet/data.csv")
)
```

### Option 3: Use Kaggle Dataset

1. **Download from Kaggle:**
   - Search for "Fake News" or "Misinformation"
   - Download dataset

2. **Load into project:**
```python
df = pd.read_csv("path/to/kaggle_dataset.csv")
```

---

## 📝 Data Format Requirements

Your dataset should have these columns (minimum):

**Required:**
- `text`: News article or post text
- `label`: Binary label (0=real, 1=fake)

**Recommended:**
- `user_id`: User identifier
- `timestamp`: Post timestamp
- `post_id`: Unique post identifier

**Optional (for enhanced features):**
- `follower_count`: Number of followers
- `following_count`: Number of following
- `verified`: Verification status
- `retweet_count`: Retweet count
- `like_count`: Like count
- `original_user_id`: For retweet networks

---

## 🔧 Adapting the Project for Your Dataset

The project is designed to be flexible. To use your dataset:

1. **Ensure required columns exist** (text, label)
2. **Map your column names** to expected names in `config.py`
3. **Or modify the preprocessing functions** to match your format

Example:
```python
# If your dataset has different column names
df = df.rename(columns={
    'article_text': 'text',
    'is_fake': 'label',
    'author_id': 'user_id'
})
```

---

## 📚 Additional Resources

- **FakeNewsNet Paper**: https://arxiv.org/abs/1809.01286
- **NELA-GT-2020 Paper**: https://arxiv.org/abs/2102.04567
- **CoAID Paper**: Check their GitHub repository
- **Dataset Collection**: https://github.com/sumeetkr/AwesomeFakeNews

---

## ❓ Still Not Sure?

**If you want social network analysis**: → **FakeNewsNet**

**If you want large-scale training**: → **NELA-GT-2020**

**If you want quick testing**: → **Use sample data function**

**If you want health misinformation**: → **CoAID**

**If you want easiest setup**: → **Kaggle Fake News dataset**

---

**Recommendation**: Start with **FakeNewsNet** for the best social network analysis experience, or use the **sample data function** to test the pipeline first!

