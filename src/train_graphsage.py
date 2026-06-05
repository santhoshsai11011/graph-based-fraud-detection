import torch

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


def train_graphsage(
    model,
    data,
    epochs=100
):

    fraud_count = (
        data.y[
            data.train_mask
        ] == 1
    ).sum()

    legit_count = (
        data.y[
            data.train_mask
        ] == 0
    ).sum()

    class_weights = torch.tensor(
        [
            1.0,
            legit_count / fraud_count
        ],
        dtype=torch.float
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    for epoch in range(
        epochs
    ):

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index
        )

        loss = criterion(
            out[
                data.train_mask
            ],
            data.y[
                data.train_mask
            ]
        )

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch}"
                f" | Loss: {loss:.4f}"
            )

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        y_true = (
            data.y[
                data.test_mask
            ]
            .cpu()
            .numpy()
        )

        y_pred = (
            predictions[
                data.test_mask
            ]
            .cpu()
            .numpy()
        )

    print(
        "\nGraphSAGE Classification Report"
    )

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    return {
        "precision":
            precision_score(
                y_true,
                y_pred
            ),

        "recall":
            recall_score(
                y_true,
                y_pred
            ),

        "f1":
            f1_score(
                y_true,
                y_pred
            )
    }