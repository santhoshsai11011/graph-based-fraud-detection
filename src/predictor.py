import joblib


class FraudPredictor:

    def __init__(self):

        self.model = joblib.load(
            "models/xgboost_model.pkl"
        )

    def predict_transaction(
        self,
        transaction_row
    ):

        drop_columns = [
            "tx_id",
            "class",
            "target"
        ]

        X = transaction_row.drop(
            columns=drop_columns,
            errors="ignore"
        )

        probability = (
            self.model
            .predict_proba(X)[0][1]
        )

        prediction = (
            self.model
            .predict(X)[0]
        )

        risk = "LOW"

        if probability > 0.90:
            risk = "HIGH"

        elif probability > 0.70:
            risk = "MEDIUM"

        return {
            "probability": float(probability),
            "prediction": int(prediction),
            "risk": risk
        }