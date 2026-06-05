def rename_feature_columns(features_df):
    """
    Column 0 -> Transaction ID
    Column 1 -> Time Step
    Columns 2-166 -> Features
    """

    num_cols = features_df.shape[1]

    columns = ["tx_id", "time_step"]

    for i in range(2, num_cols):
        columns.append(f"feature_{i-1}")

    features_df.columns = columns

    return features_df


def merge_features_and_labels(features_df, classes_df):
    merged_df = features_df.merge(
        classes_df,
        left_on="tx_id",
        right_on="txId",
        how="left"
    )

    # Remove duplicate transaction id column
    merged_df.drop(columns=["txId"], inplace=True)

    return merged_df


def validate_data(df):
    print("\nValidation Report")
    print("-" * 40)

    print(f"Total Rows: {len(df)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate tx_ids: {df['tx_id'].duplicated().sum()}")

    print("\nClass Distribution:")
    print(df["class"].value_counts())

    print("-" * 40)


def preprocess_data(features_df, classes_df):
    features_df = rename_feature_columns(features_df)

    merged_df = merge_features_and_labels(
        features_df,
        classes_df
    )

    return merged_df