import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predictor import FraudPredictor
from network_visualization import build_transaction_network
from explainability import get_feature_importance

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Graph-Based Fraud Detection", page_icon="🔍", layout="wide")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/graph_feature_dataset.csv")

@st.cache_data
def load_edges():
    return pd.read_csv("data/raw/elliptic_txs_edgelist.csv")

dataset = load_dataset()
edges = load_edges()
predictor = FraudPredictor()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🔍 Graph-Based Fraud Detection System")
st.markdown(
    """
Detect suspicious Bitcoin transactions using machine learning,
graph analytics and graph neural networks.
"""
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Transactions", "203,769")
with col2:
    st.metric("Edges", "234,355")
with col3:
    st.metric("Labeled Transactions", "37,594")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.subheader("🏆 Model Performance")
performance_df = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "GraphSAGE", "XGBoost"],
        "F1 Score": [0.353, 0.712, 0.845],
    }
)
st.dataframe(performance_df, width="stretch")
st.bar_chart(performance_df.set_index("Model"))

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Transaction Lookup")
transaction_id = st.sidebar.text_input("Enter Transaction ID")
st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")
st.sidebar.write("Dataset: Elliptic Bitcoin Dataset")
st.sidebar.write("Best Model: XGBoost")
st.sidebar.write("F1 Score: 0.845")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.subheader("🕵️ Transaction Fraud Detection")

if st.sidebar.button("Predict"):
    try:
        transaction_id = int(transaction_id)
        transaction = dataset[dataset["tx_id"] == transaction_id]

        if transaction.empty:
            st.error("Transaction not found.")
        else:
            result = predictor.predict_transaction(transaction)

            actual_class = str(transaction["class"].iloc[0])
            if actual_class == "1":
                actual_label = "FRAUD"
            elif actual_class == "2":
                actual_label = "LEGITIMATE"
            else:
                actual_label = "UNKNOWN"

            prediction = "FRAUD" if result["prediction"] == 1 else "LEGITIMATE"

            st.subheader("Prediction Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability", f"{result['probability']:.2%}")
            m2.metric("Risk Level", result["risk"])
            m3.metric("Prediction", prediction)

            st.markdown("---")
            st.subheader("Transaction Information")
            c1, c2 = st.columns(2)
            c1.write(f"**Transaction ID:** {transaction_id}")
            c2.write(f"**Actual Label:** {actual_label}")

            st.markdown("---")
            st.subheader("Transaction Record")
            st.dataframe(transaction, width="stretch")

            st.markdown("---")
            st.subheader("🔍 Fraud Investigation Insights")
            feature_importance = get_feature_importance(predictor.get_model())
            st.write("Top Features Used By XGBoost")
            st.dataframe(feature_importance, width="stretch")
            st.bar_chart(feature_importance.set_index("Feature"))

            st.markdown("---")
            st.subheader("📈 Transaction Graph Metrics")
            g1, g2, g3 = st.columns(3)
            g1.metric("Degree", int(transaction["degree"].iloc[0]))
            g2.metric("PageRank", round(float(transaction["pagerank"].iloc[0]), 6))
            g3.metric("Clustering Coefficient", round(float(transaction["clustering_coef"].iloc[0]), 4))

            st.markdown("---")
            st.subheader("🌐 Transaction Network")
            html_file = build_transaction_network(transaction_id, edges, dataset)
            with open(html_file, "r", encoding="utf-8") as file:
                html = file.read()
            components.html(html, height=750)

    except Exception as e:
        st.error(str(e))