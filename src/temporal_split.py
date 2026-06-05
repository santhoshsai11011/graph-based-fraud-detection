def temporal_split(df):

    train_df = df[
        df["time_step"] <= 34
    ]

    val_df = df[
        (df["time_step"] >= 35)
        &
        (df["time_step"] <= 39)
    ]

    test_df = df[
        df["time_step"] >= 40
    ]

    return (
        train_df,
        val_df,
        test_df
    )