from load_data import load_all_data
from preprocess import preprocess_data, validate_data


def main():
    print("\nLoading Dataset...\n")

    features, classes, edges = load_all_data()

    processed_df = preprocess_data(
        features,
        classes
    )

    print("Dataset Loaded Successfully\n")

    print(f"Features Shape: {features.shape}")
    print(f"Classes Shape: {classes.shape}")
    print(f"Edges Shape: {edges.shape}")

    print(f"\nProcessed Dataset Shape: {processed_df.shape}")

    validate_data(processed_df)

    print("\nSample Rows:")
    print(processed_df.head())


if __name__ == "__main__":
    main()