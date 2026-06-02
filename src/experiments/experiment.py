import numpy as np
import torch

from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import train
from training.evaluation import get_errors, metrics
from utils.visualization import plot_results

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest


def run_experiment(df, params, device):

    print("\n================ RUN =================")
    print(params)

    # ---- WINDOWING INPUT ----
    from data.windowing import make_windows, split_windows

    x, y = make_windows(
        df["value"].values,
        df["label"].values,
        params["seq_len"]
    )

    tr_idx, te_idx = split_windows(x, y)

    x_tr, x_te = x[tr_idx], x[te_idx]
    y_te = y[te_idx]

    # ---- MODEL ----
    model = LSTMAutoencoder(
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        num_layers=params["num_layers"]
    )

    model, _ = train(
        model,
        x_tr,
        {
            "epochs": 50,
            "batch_size": 64,
            "lr": 1e-3
        },
        device
    )

    # ---- RECONSTRUCTION SCORES ----
    tr_scores = get_errors(model, x_tr, device)
    te_scores = get_errors(model, x_te, device)

    # ---- THRESHOLD SEARCH ----
    best_f1 = -1
    best_thr = 0
    best_p = None

    for p in np.arange(30, 100, 1):

        thr = np.percentile(tr_scores, p)

        p_, r_, f1_ = metrics(y_te, te_scores, thr)

        if f1_ > best_f1:
            best_f1 = f1_
            best_thr = thr
            best_p = p

    # ---- FINAL METRICS ----
    p, r, f1 = metrics(y_te, te_scores, best_thr)

    roc_auc = roc_auc_score(y_te, te_scores)
    pr_auc = average_precision_score(y_te, te_scores)

    # ---- ISOLATION FOREST BASELINE ----
    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    x_tr_flat = x_tr.reshape(len(x_tr), -1)
    x_te_flat = x_te.reshape(len(x_te), -1)

    iso.fit(x_tr_flat)

    iso_scores = -iso.decision_function(x_te_flat)

    iso_pred = (iso.predict(x_te_flat) == -1).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score

    iso_p = precision_score(y_te, iso_pred, zero_division=0)
    iso_r = recall_score(y_te, iso_pred, zero_division=0)
    iso_f1 = f1_score(y_te, iso_pred, zero_division=0)

    iso_roc = roc_auc_score(y_te, iso_scores)
    iso_pr = average_precision_score(y_te, iso_scores)

    # ---- VISUALIZATION ----
    plot_results(
        df["value"].values[:len(y_te)],
        y_te,
        te_scores,
        best_thr,
        title=f"Window={params['seq_len']}"
    )

    # ---- RETURN RESULTS ----
    return {
        "hidden_dim": params["hidden_dim"],
        "latent_dim": params["latent_dim"],
        "num_layers": params["num_layers"],
        "seq_len": params["seq_len"],

        "threshold": best_thr,
        "best_percentile": best_p,

        "precision": p,
        "recall": r,
        "f1": f1,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "if_precision": iso_p,
        "if_recall": iso_r,
        "if_f1": iso_f1,
        "if_roc_auc": iso_roc,
        "if_pr_auc": iso_pr
    }