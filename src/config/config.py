import numpy as np

CONFIG = {

    # ---- MODEL SEARCH SPACE ----
    "search_space": {
        "hidden_dim": [128],
        "latent_dim": [4],
        "num_layers": [1],
        "seq_len": np.arange(64, 352, 32)
    },

    # ---- TRAINING ----
    "train": {
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "early_stopping_patience": 10,
        "scheduler_patience": 3,
        "scheduler_factor": 0.5,
        "min_lr": 1e-6
    },

    # ---- THRESHOLD SEARCH ----
    "threshold_percentiles": np.arange(30, 100, 1)
}