from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

import joblib


def train_xgboost_temporal(
    train_df,
    test_df
):

    drop_columns = [
        "tx_id",
        "class",
        "target"
    ]

    X_train = train_df.drop(
        columns=drop_columns
    )

    y_train = train_df["target"]

    X_test = test_df.drop(
        columns=drop_columns
    )

    y_test = test_df["target"]

    fraud_count = sum(
        y_train == 1
    )

    legit_count = sum(
        y_train == 0
    )

    scale_pos_weight = (
        legit_count /
        fraud_count
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    results = {
        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions
            ),

        "recall":
            recall_score(
                y_test,
                predictions
            ),

        "f1":
            f1_score(
                y_test,
                predictions
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            )
    }

    print(
        "\nClassification Report"
    )

    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        "\nConfusion Matrix"
    )

    print("-" * 50)

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return (
        model,
        results,
        X_train.columns
    )


def save_xgboost(
    model
):

    joblib.dump(
        model,
        "models/xgboost_model.pkl"
    )