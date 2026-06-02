# ---- IMPORTS ---- #
import numpy as np


# ---- MAKE WINDOWS ---- #
def make_windows(values, labels, seq_len):

    # ---- INPUT PREPROCESSING ---- #
    values = np.asarray(values).reshape(-1, 1)
    labels = np.asarray(labels)

    # ---- WINDOW COUNT ---- #
    n = len(values) - seq_len + 1

    # ---- ALLOCATE ARRAYS ---- #
    x = np.zeros((n, seq_len, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    # ---- WINDOW CONSTRUCTION ---- #
    for i in range(n):

        # ---- SEQUENCE SLICE ---- #
        x[i] = values[i:i + seq_len]

        # ---- LABEL AGGREGATION ---- #
        y[i] = np.max(labels[i:i + seq_len])

    return x, y


# ---- SPLIT WINDOWS ---- #
def split_windows(x, y):

    # ---- TRAIN INDEX (NORMAL ONLY) ---- #
    train_idx = np.where(y == 0)[0]

    # ---- TEST INDEX (ALL WINDOWS) ---- #
    test_idx = np.arange(len(y))

    # ---- SHUFFLE TRAIN SET ---- #
    np.random.shuffle(train_idx)

    return train_idx, test_idx