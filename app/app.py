import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Import from src folder
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent / "src"
    )
)

from predictor import FraudPredictor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Graph-Based Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/processed/graph_feature_dataset.csv"
    )


dataset = load_dataset()

predictor = FraudPredictor()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🔍 Graph-Based Fraud Detection System"
)

st.markdown(
    """
Detect suspicious Bitcoin transactions using machine learning,
graph analytics, and network topology features.
"""
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader(
    "📊 Dataset Overview"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Transactions",
        "203,769"
    )

with col2:
    st.metric(
        "Edges",
        "234,355"
    )

with col3:
    st.metric(
        "Labeled Transactions",
        "37,594"
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader(
    "🏆 Model Performance"
)

performance_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "GraphSAGE",
            "XGBoost"
        ],
        "F1 Score": [
            0.353,
            0.712,
            0.845
        ]
    }
)

st.dataframe(
    performance_df,
    use_container_width=True
)

st.bar_chart(
    performance_df.set_index(
        "Model"
    )
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "Transaction Lookup"
)

transaction_id = st.sidebar.text_input(
    "Enter Transaction ID"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Project Information"
)

st.sidebar.write(
    "Dataset: Elliptic Bitcoin Dataset"
)

st.sidebar.write(
    "Best Model: XGBoost"
)

st.sidebar.write(
    "F1 Score: 0.845"
)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.subheader(
    "🕵️ Transaction Fraud Detection"
)

if st.sidebar.button(
    "Predict"
):

    if not transaction_id:

        st.warning(
            "Please enter a Transaction ID."
        )

    else:

        try:

            transaction_id = int(
                transaction_id
            )

            transaction = dataset[
                dataset["tx_id"]
                == transaction_id
            ]

            if transaction.empty:

                st.error(
                    "Transaction not found."
                )

            else:

                result = (
                    predictor
                    .predict_transaction(
                        transaction
                    )
                )

                actual_class = str(
                    transaction["class"]
                    .iloc[0]
                )

                if actual_class == "1":

                    actual_label = "FRAUD"

                elif actual_class == "2":

                    actual_label = "LEGITIMATE"

                else:

                    actual_label = "UNKNOWN"

                if result["prediction"] == 1:

                    prediction = "FRAUD"

                else:

                    prediction = "LEGITIMATE"

                st.subheader(
                    "Prediction Results"
                )

                metric1, metric2, metric3 = (
                    st.columns(3)
                )

                metric1.metric(
                    "Fraud Probability",
                    f"{result['probability']:.2%}"
                )

                metric2.metric(
                    "Risk Level",
                    result["risk"]
                )

                metric3.metric(
                    "Prediction",
                    prediction
                )

                st.markdown("---")

                st.subheader(
                    "Transaction Information"
                )

                info1, info2 = st.columns(2)

                info1.write(
                    f"**Transaction ID:** {transaction_id}"
                )

                info2.write(
                    f"**Actual Label:** {actual_label}"
                )

                st.markdown("---")

                st.subheader(
                    "Transaction Record"
                )

                st.dataframe(
                    transaction,
                    use_container_width=True
                )

        except ValueError:

            st.error(
                "Please enter a valid numeric Transaction ID."
            )