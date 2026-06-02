# ---- IMPORTS ---- #
import numpy as np
import torch

from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import train
from training.evaluation import get_errors, metrics
from utils.visualization import plot_results
from data.windowing import make_windows, split_windows

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import seaborn as sns


# ---- RUN EXPERIMENT ---- #
def run_experiment(df, params, device):

    print("\n================ RUN =================")
    print(params)

    # ---- WINDOWING ---- #
    x, y = make_windows(
        df["value"].values,
        df["label"].values,
        params["seq_len"]
    )

    # ---- SPLIT ---- #
    tr_idx, te_idx = split_windows(x, y)

    x_tr, x_te = x[tr_idx], x[te_idx]
    y_te = y[te_idx]

    # ---- MODEL ---- #
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

    # ---- SCORING ---- #
    tr_scores = get_errors(model, x_tr, device)
    te_scores = get_errors(model, x_te, device)

    # ---- THRESHOLD SEARCH ---- #
    print("\n===== THRESHOLD SEARCH =====")

    best_thr = None
    best_f1 = -1
    best_percentile = None

    for p in range(1, 100):

        thr = np.percentile(tr_scores, p)

        _, _, f1_v = metrics(y_te, te_scores, thr)

        print(
            f"p={p:2d}% | "
            f"thr={thr:.6f} | "
            f"F1={f1_v:.4f}"
        )

        if f1_v > best_f1:
            best_f1 = f1_v
            best_thr = thr
            best_percentile = p

    print("\nBEST THRESHOLD:", best_thr)

    # ---- FINAL METRICS ---- #
    precision, recall, f1 = metrics(y_te, te_scores, best_thr)

    roc_auc = roc_auc_score(y_te, te_scores)
    pr_auc = average_precision_score(y_te, te_scores)

    print(
        f"\nTEST -> "
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC_AUC={roc_auc:.4f} "
        f"PR_AUC={pr_auc:.4f}"
    )

    # ---- CONFUSION MATRIX ---- #
    y_pred = (te_scores > best_thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    print("\n===== CONFUSION MATRIX =====")
    print(f"TN={tn}  FP={fp}")
    print(f"FN={fn}  TP={tp}")

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (seq_len={params['seq_len']})")
    plt.show()

    # ---- ISOLATION FOREST ---- #
    print("\n===== ISOLATION FOREST =====")

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    x_tr_flat = x_tr.reshape(len(x_tr), -1)
    x_te_flat = x_te.reshape(len(x_te), -1)

    iso.fit(x_tr_flat)

    iso_scores = -iso.decision_function(x_te_flat)

    iso_pred = (iso.predict(x_te_flat) == -1).astype(int)

    iso_precision = precision_score(y_te, iso_pred, zero_division=0)
    iso_recall = recall_score(y_te, iso_pred, zero_division=0)
    iso_f1 = f1_score(y_te, iso_pred, zero_division=0)

    iso_roc_auc = roc_auc_score(y_te, iso_scores)
    iso_pr_auc = average_precision_score(y_te, iso_scores)

    print(
        f"IF -> "
        f"P={iso_precision:.4f} "
        f"R={iso_recall:.4f} "
        f"F1={iso_f1:.4f} "
        f"ROC_AUC={iso_roc_auc:.4f} "
        f"PR_AUC={iso_pr_auc:.4f}"
    )

    # ---- VISUALIZATION ---- #
    plot_results(
        x_te[:, 0] if len(x_te.shape) > 1 else x_te,
        y_te,
        te_scores,
        best_thr,
        title=f"seq_len={params['seq_len']}"
    )

    # ---- RETURN ---- #
    return {
        "hidden_dim": params["hidden_dim"],
        "latent_dim": params["latent_dim"],
        "num_layers": params["num_layers"],
        "seq_len": params["seq_len"],

        "threshold": best_thr,
        "best_percentile": best_percentile,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "if_precision": iso_precision,
        "if_recall": iso_recall,
        "if_f1": iso_f1,
        "if_roc_auc": iso_roc_auc,
        "if_pr_auc": iso_pr_auc
    }