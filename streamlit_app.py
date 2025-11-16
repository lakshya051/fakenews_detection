
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config


@st.cache_resource
def load_model():
    """Load trained model (cached as resource, not data). Import lazily to avoid torch import issues."""
    from src.models import TraditionalMLModel

    model_path = config.TRAINING_CONFIG["model_save_dir"] / "fakenewsnet_model.pkl"
    model_wrapper = TraditionalMLModel("random_forest")
    try:
        model_wrapper.load(model_path)
        st.success("✓ Model loaded successfully")
    except Exception as e:
        st.error(f"Could not load model from {model_path}: {e}")
        raise
    return model_wrapper


@st.cache_resource
def get_extractor():
    """Initialize feature extractor (cached as resource). Import lazily to avoid torch/transformers import during Streamlit inspection."""
    from src.feature_extractor import FeatureExtractor
    return FeatureExtractor(use_bert=False)


def build_input_df(title: str,
                   user_id: str = "demo_user",
                   timestamp: str = None,
                   follower_count: int = 100,
                   following_count: int = 50,
                   verified: int = 0,
                   account_created: str = None) -> pd.DataFrame:
    # Prepare a one-row DataFrame matching pipeline expectations
    if timestamp is None:
        timestamp = pd.Timestamp.now().isoformat()
    if account_created is None:
        account_created = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()

    row = {
        "title": title,
        "user_id": user_id,
        "timestamp": timestamp,
        "follower_count": follower_count,
        "following_count": following_count,
        "verified": verified,
        "account_created": account_created,
    }
    return pd.DataFrame([row])


def predict_single(title: str, model, extractor, **kwargs):
    """Predict on a single text with metadata."""
    df = build_input_df(title, **kwargs)
    
    try:
        features = extractor.extract_all_features(
            df, text_column="title", user_column="user_id", timestamp_column="timestamp"
        )
    except Exception as e:
        st.warning(f"Feature extraction failed: {e}")
        return None
    
    if features.empty:
        st.warning("No features extracted for input; check input fields.")
        return None

    X = features.values
    try:
        pred = model.predict(X)[0]
        try:
            proba = model.predict_proba(X)[0]
        except:
            proba = None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

    return {
        "prediction": int(pred),
        "proba": proba,
        "features": features
    }


def batch_predict(df: pd.DataFrame, model, extractor, text_column: str = "title") -> pd.DataFrame:
    """Batch predict on multiple rows. Auto-fills missing metadata columns."""
    df = df.copy()
    
    # Auto-fill missing metadata columns with defaults
    defaults = {
        "user_id": "batch_user",
        "timestamp": pd.Timestamp.now().isoformat(),
        "follower_count": 100,
        "following_count": 50,
        "verified": 0,
        "account_created": (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    try:
        features = extractor.extract_all_features(
            df, text_column=text_column, user_column="user_id", timestamp_column="timestamp"
        )
    except Exception as e:
        st.warning(f"Feature extraction failed: {e}")
        return pd.DataFrame()
    
    if features.empty:
        st.warning("Feature extraction returned empty DataFrame for batch input.")
        return pd.DataFrame()

    X = features.values
    try:
        preds = model.predict(X)
        try:
            proba = model.predict_proba(X)
        except:
            proba = None
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
        return pd.DataFrame()

    results = df[[text_column, "user_id"]].copy()
    results["prediction"] = preds
    results["predicted_label"] = results["prediction"].map({0: "Real", 1: "Fake"})
    
    if proba is not None and proba.shape[1] >= 2:
        results["prob_real"] = proba[:, 0]
        results["prob_fake"] = proba[:, 1]

    return results


def main():
    st.set_page_config(page_title="FakeNews Detection Demo", layout="wide")
    st.title("FakeNews Detection — Demo Streamlit App")

    st.markdown("Use the sidebar to enter text or upload a CSV for batch prediction.")

    # Load resources
    model = load_model()
    extractor = get_extractor()

    st.sidebar.header("Prediction Settings")
    mode = st.sidebar.radio("Mode", ["Single", "Batch"])

    if mode == "Single":
        st.subheader("Single text prediction")
        title = st.text_area("Enter article/title text", height=150, value="Enter a news headline or short article text here.")

        st.sidebar.subheader("Optional metadata")
        follower_count = st.sidebar.number_input("Follower count", value=100, min_value=0)
        following_count = st.sidebar.number_input("Following count", value=50, min_value=0)
        verified = st.sidebar.checkbox("Verified user", value=False)
        account_created = st.sidebar.date_input("Account created", value=(datetime.date.today() - datetime.timedelta(days=365)))
        user_id = st.sidebar.text_input("User ID", value="demo_user")

        if st.button("Predict"):
            with st.spinner("Extracting features and running model..."):
                try:
                    result = predict_single(
                        title,
                        model=model,
                        extractor=extractor,
                        user_id=user_id,
                        timestamp=pd.Timestamp.now().isoformat(),
                        follower_count=int(follower_count),
                        following_count=int(following_count),
                        verified=int(verified),
                        account_created=pd.Timestamp(account_created).isoformat()
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            if result is None:
                return

            label = "Fake" if result["prediction"] == 1 else "Real"
            st.metric("Prediction", label)
            if result["proba"] is not None:
                st.subheader("Probabilities")
                proba = result["proba"]
                for i, p in enumerate(proba):
                    st.write(f"Class {i}: {p:.4f}")
                st.bar_chart(pd.DataFrame({'prob': proba, 'class': list(range(len(proba)))}).set_index('class'))

            st.subheader("Extracted features (preview)")
            st.dataframe(result["features"])

    else:
        st.subheader("Batch prediction (CSV)")
        uploaded = st.file_uploader("Upload CSV with a `title` column", type=["csv"])
        sample_button = st.button("Load demo samples")

        if sample_button and not uploaded:
            demo_path = Path("data") / "processed" / "demo_samples.csv"
            try:
                uploaded_df = pd.read_csv(demo_path)
            except Exception as e:
                st.error(f"Could not load demo samples: {e}")
                uploaded_df = None
        elif uploaded:
            uploaded_df = pd.read_csv(uploaded)
        else:
            uploaded_df = None

        if uploaded_df is not None:
            st.write(f"Loaded {len(uploaded_df)} rows")
            with st.spinner("Running batch predictions..."):
                results = batch_predict(uploaded_df, model=model, extractor=extractor, text_column="title")
            if not results.empty:
                st.dataframe(results.head(200))
                csv = results.to_csv(index=False)
                st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
