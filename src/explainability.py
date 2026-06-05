import pandas as pd


def get_feature_importance(
    model,
    top_n=15
):

    importance_df = pd.DataFrame(
        {
            "Feature":
                model.feature_names_in_,

            "Importance":
                model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(top_n)
    )

    return importance_df