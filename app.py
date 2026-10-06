import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Groceries Clustering", layout="wide")

# ------------------------------------------------------------------
# Raw GitHub URL — always available, no local file needed
# ------------------------------------------------------------------
RAW_DATASET_URL = (
    "https://raw.githubusercontent.com/Anwaarkhan1728/"
    "Groceries-Data-Mining-Clustering-Association-Rule-Discovery-"
    "for-Retail-Intelligence/main/Groceries_dataset.csv"
)

# ------------------------------------------------------------------
# Helper: find a file among multiple possible names / locations
# ------------------------------------------------------------------
def find_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

SCALER_CANDIDATES = [
    "scaler.pkl", "scaler (1).pkl",
]
MODEL_CANDIDATES = [
    "best_clustering_model.pkl",
]
FEATURES_CANDIDATES = [
    "feature_columns.pkl",
]
MODELNAME_CANDIDATES = [
    "best_model_name.pkl",
]
SUBMISSION_CANDIDATES = [
    "submission.csv", "submission (1).csv",
]
# NOTE: The raw GitHub URL is included as a fallback for the dataset
DATASET_CANDIDATES = [
    "Groceries_dataset.csv",
    RAW_DATASET_URL,
]

# ------------------------------------------------------------------
# Safe loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {
        "scaler": None, "model": None,
        "feature_cols": None, "model_name": None,
        "errors": []
    }

    p = find_file(SCALER_CANDIDATES)
    if p:
        try:
            artifacts["scaler"] = joblib.load(p)
        except Exception as e:
            artifacts["errors"].append(f"scaler load failed: {e}")
    else:
        artifacts["errors"].append(
            "scaler.pkl not found (rename 'scaler (1).pkl' → 'scaler.pkl')"
        )

    p = find_file(MODEL_CANDIDATES)
    if p:
        try:
            artifacts["model"] = joblib.load(p)
        except Exception as e:
            artifacts["errors"].append(f"model load failed: {e}")
    else:
        artifacts["errors"].append("best_clustering_model.pkl not found")

    p = find_file(FEATURES_CANDIDATES)
    if p:
        try:
            artifacts["feature_cols"] = joblib.load(p)
        except Exception as e:
            artifacts["errors"].append(f"feature_columns load failed: {e}")
    else:
        artifacts["errors"].append("feature_columns.pkl not found")

    p = find_file(MODELNAME_CANDIDATES)
    if p:
        try:
            artifacts["model_name"] = joblib.load(p)
        except Exception as e:
            artifacts["errors"].append(f"model_name load failed: {e}")
    else:
        artifacts["errors"].append("best_model_name.pkl not found")

    return artifacts


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df, member_features, errors = None, None, []

    # --- submission.csv (local only) ---
    p = find_file(SUBMISSION_CANDIDATES)
    if p:
        try:
            member_features = pd.read_csv(p)
        except Exception as e:
            errors.append(f"submission load failed: {e}")
    else:
        errors.append(
            "submission.csv not found (rename 'submission (1).csv' → 'submission.csv')"
        )

    # --- raw dataset: try local file first, then raw GitHub URL ---
    for source in DATASET_CANDIDATES:
        try:
            if source.startswith("http"):
                df = pd.read_csv(source)
            else:
                if os.path.exists(source):
                    df = pd.read_csv(source)
                else:
                    continue
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            errors.append(f"dataset load failed ({source[:60]}…): {e}")

    if df is None:
        errors.append(
            "Groceries_dataset.csv could not be loaded from local repo "
            "or raw GitHub URL."
        )

    return df, member_features, errors


artifacts = load_artifacts()
df, member_features, data_errors = load_data()

scaler       = artifacts["scaler"]
model        = artifacts["model"]
feature_cols = artifacts["feature_cols"]
model_name   = artifacts["model_name"]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🛒 Groceries Customer Clustering")
st.markdown("Unsupervised Machine Learning on Grocery Transactions")

# Diagnostics panel
with st.sidebar.expander(
    "🔧 Diagnostics",
    expanded=bool(artifacts["errors"] or data_errors)
):
    st.write("**Artifacts:**")
    st.write(f"- scaler: {'✅' if scaler is not None else '❌'}")
    st.write(f"- model: {'✅' if model is not None else '❌'}")
    st.write(f"- feature_cols: {'✅' if feature_cols is not None else '❌'}")
    st.write(f"- model_name: {'✅' if model_name is not None else '❌'}")
    st.write(f"- submission df: {'✅' if member_features is not None else '❌'}")
    st.write(f"- raw dataset: {'✅' if df is not None else '❌'}")
    for e in artifacts["errors"] + data_errors:
        st.warning(e)

# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------
pages = ["Overview", "EDA", "Clusters", "Predict"]
page = st.sidebar.radio("Go to", pages)

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------
if page == "Overview":
    st.header("Dataset Overview")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Unique Members", df["Member_number"].nunique())
        c3.metric("Unique Items", df["itemDescription"].nunique())
        st.dataframe(df.head(100))
    else:
        st.error("Raw dataset not loaded from any source.")

    if model_name is not None:
        st.success(f"Best Clustering Model: **{model_name}**")

# ------------------------------------------------------------------
# EDA
# ------------------------------------------------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    if df is None:
        st.error("Raw dataset not loaded.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 15 Items")
            top_items = df["itemDescription"].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            top_items.plot(kind="barh", ax=ax, color="steelblue")
            ax.invert_yaxis()
            st.pyplot(fig)
        with c2:
            st.subheader("Basket Size Distribution")
            basket_sizes = (
                df.groupby(["Member_number", "Date"])["itemDescription"].count()
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(basket_sizes, bins=20, color="coral", edgecolor="black")
            ax.set_xlabel("Items per Transaction")
            st.pyplot(fig)

# ------------------------------------------------------------------
# Clusters
# ------------------------------------------------------------------
elif page == "Clusters":
    st.header("Cluster Analysis")
    if member_features is None:
        st.error("submission.csv not loaded.")
    else:
        st.subheader("Cluster Distribution")
        cluster_counts = member_features["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        profile_cols = [
            c for c in member_features.columns
            if c not in ["Member_number", "Cluster"]
        ]
        st.subheader("Cluster Profiles (mean)")
        st.dataframe(
            member_features.groupby("Cluster")[profile_cols].mean()
        )

        if scaler is not None and len(profile_cols) > 1:
            st.subheader("Cluster Visualization (PCA)")
            X = (
                member_features[profile_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            X_scaled = scaler.transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=member_features["Cluster"],
                cmap="viridis", alpha=0.6, s=10,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(sc, ax=ax, label="Cluster")
            st.pyplot(fig)

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
elif page == "Predict":
    st.header("Predict Cluster for New Member")
    if model is None or scaler is None or feature_cols is None:
        st.error("Model/scaler/feature_columns not loaded — cannot predict.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            total_items = st.number_input(
                "Total Items Purchased", min_value=0, value=50
            )
            unique_items = st.number_input(
                "Unique Items Purchased", min_value=0, value=20
            )
            num_transactions = st.number_input(
                "Number of Transactions", min_value=1, value=5
            )
        with c2:
            avg_basket_size = st.number_input(
                "Average Basket Size", min_value=0.0, value=10.0
            )

        input_data = {
            "total_items": total_items,
            "unique_items": unique_items,
            "num_transactions": num_transactions,
            "avg_basket_size": avg_basket_size,
        }
        for col in feature_cols:
            if col not in input_data:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])[feature_cols]

        if st.button("Predict Cluster"):
            X_scaled = scaler.transform(input_df)
            cluster = int(model.predict(X_scaled)[0])
            st.success(f"Predicted Cluster: **{cluster}**")
            st.balloons()

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | Groceries Clustering Project")
