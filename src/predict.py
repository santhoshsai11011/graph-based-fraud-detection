import pandas as pd

from predictor import FraudPredictor


dataset = pd.read_csv(
    "data/processed/graph_feature_dataset.csv"
)

predictor = FraudPredictor()

fraud_samples = dataset[
    dataset["class"] == "1"
]

print(
    fraud_samples["tx_id"]
    .head()
)


transaction_id = int(
    input(
        "\nEnter Transaction ID: "
    )
)

transaction = dataset[
    dataset["tx_id"] == transaction_id
]

if transaction.empty:

    print(
        "\nTransaction Not Found"
    )

    exit()


result = predictor.predict_transaction(
    transaction
)

actual_class = transaction["class"].iloc[0]

if actual_class == "1":
    actual_label = "FRAUD"

elif actual_class == "2":
    actual_label = "LEGITIMATE"

else:
    actual_label = "UNKNOWN"

print(
    f"Actual Label: {actual_label}"
)

print(
    "\nPrediction Result"
)

print("-" * 40)

print(
    f"Transaction ID: {transaction_id}"
)

print(
    f"Fraud Probability: "
    f"{result['probability']:.4f}"
)

print(
    f"Risk Level: "
    f"{result['risk']}"
)

if result["prediction"] == 1:

    print(
        "Prediction: FRAUD"
    )

else:

    print(
        "Prediction: LEGITIMATE"
    )