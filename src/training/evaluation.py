# ---- IMPORTS ---- #
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


# ---- ERROR COMPUTATION ---- #
def get_errors(model, x, device):

    # ---- MODEL EVAL MODE ---- #
    model.eval()

    # ---- DATALOADER ---- #
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x)),
        batch_size=256
    )

    errs = []

    # ---- INFERENCE ---- #
    with torch.no_grad():
        for (b,) in loader:

            b = b.to(device)
            r = model(b)

            e = torch.mean((r - b) ** 2, dim=(1, 2))
            errs.append(e.cpu())

    return torch.cat(errs).numpy()


# ---- CLASSIFICATION METRICS ---- #
def metrics(y_true, scores, thr):

    # ---- BINARY PREDICTION ---- #
    pred = (scores > thr).astype(int)

    return (
        precision_score(y_true, pred, zero_division=0),
        recall_score(y_true, pred, zero_division=0),
        f1_score(y_true, pred, zero_division=0)
    )