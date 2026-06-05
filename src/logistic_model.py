from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib


def train_logistic_temporal(
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

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
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
        scaler,
        results
    )


def save_model(
    model,
    scaler
):

    joblib.dump(
        model,
        "models/logistic_regression.pkl"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )