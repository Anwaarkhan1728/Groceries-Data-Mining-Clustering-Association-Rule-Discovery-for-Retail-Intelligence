import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="Groceries Clustering", layout="wide")

st.title("🛒 Groceries Customer Clustering")
st.markdown("Unsupervised Machine Learning on Grocery Transactions")

# Load artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_clustering_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    model_name = joblib.load('best_model_name.pkl')
    return scaler, model, feature_cols, model_name

@st.cache_data
def load_data():
    df = pd.read_csv('/kaggle/input/datasets/anwarkhanniazi/unsupervised-association-rule/Groceries_dataset.csv')
    member_features = pd.read_csv('submission.csv')
    return df, member_features

scaler, model, feature_cols, model_name = load_artifacts()
df, member_features = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Clusters", "Association Rules", "Predict"])

# --- Page: Overview ---
if page == "Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Members", df['Member_number'].nunique())
    col3.metric("Unique Items", df['itemDescription'].nunique())
    
    st.subheader("Sample Data")
    st.dataframe(df.head(100))
    
    st.subheader("Best Model")
    st.success(f"Best Clustering Model: **{model_name}**")

# --- Page: EDA ---
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Items")
        top_items = df['itemDescription'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        top_items.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Basket Size Distribution")
        basket_sizes = df.groupby(['Member_number', 'Date'])['itemDescription'].count()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(basket_sizes, bins=20, color='coral', edgecolor='black')
        ax.set_xlabel('Items per Transaction')
        st.pyplot(fig)

# --- Page: Clusters ---
elif page == "Clusters":
    st.header("Cluster Analysis")
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = member_features['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    profile_cols = [col for col in member_features.columns if col not in ['Member_number', 'Cluster']]
    cluster_profile = member_features.groupby('Cluster')[profile_cols].mean()
    st.dataframe(cluster_profile.style.background_gradient(cmap='viridis'))
    
    # PCA visualization
    st.subheader("Cluster Visualization (PCA)")
    X = member_features[profile_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=member_features['Cluster'], cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

# --- Page: Association Rules ---
elif page == "Association Rules":
    st.header("Association Rules")
    try:
        rules = pd.read_csv('association_rules.csv')
        st.subheader("Top 20 Rules by Lift")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))
        
        st.subheader("Confidence vs Lift")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(rules['confidence'], rules['lift'], alpha=0.5, c='steelblue')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Lift')
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Association rules not found. Run Module 4 first.")

# --- Page: Predict ---
elif page == "Predict":
    st.header("Predict Cluster for New Member")
    st.markdown("Enter member statistics to predict their cluster:")
    
    col1, col2 = st.columns(2)
    with col1:
        total_items = st.number_input("Total Items Purchased", min_value=0, value=50)
        unique_items = st.number_input("Unique Items Purchased", min_value=0, value=20)
        num_transactions = st.number_input("Number of Transactions", min_value=1, value=5)
    with col2:
        avg_basket_size = st.number_input("Average Basket Size", min_value=0.0, value=10.0)
    
    # Category features (default 0 for simplicity)
    input_data = {
        'total_items': total_items,
        'unique_items': unique_items,
        'num_transactions': num_transactions,
        'avg_basket_size': avg_basket_size
    }
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0
    
    input_df = pd.DataFrame([input_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    
    if st.button("Predict Cluster"):
        cluster = model.predict(input_scaled)[0]
        st.success(f"Predicted Cluster: **{cluster}**")
        st.balloons()

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | Groceries Clustering Project")
