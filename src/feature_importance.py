import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(
    model,
    feature_names
):

    importance = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance":
                model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(20)
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.title(
        "Top 20 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/figures/xgboost_feature_importance.png"
    )

    plt.show()