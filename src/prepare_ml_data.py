import pandas as pd


def prepare_dataset(df):
    """
    Keep only labeled transactions.

    Fraud (1)
    Legitimate (2)
    """

    df = df.copy()

    df = df[
        df["class"] != "unknown"
    ]

    df["class"] = (
        df["class"]
        .astype(int)
    )

    label_map = {
        1: 1,  # Fraud
        2: 0   # Legitimate
    }

    df["target"] = (
        df["class"]
        .map(label_map)
    )

    return df