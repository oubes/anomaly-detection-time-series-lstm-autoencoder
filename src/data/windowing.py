import numpy as np


def make_windows(values, labels, seq_len):

    values = np.asarray(values).reshape(-1, 1)
    labels = np.asarray(labels)

    n = len(values) - seq_len + 1

    x = np.zeros((n, seq_len, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x[i] = values[i:i + seq_len]
        y[i] = np.max(labels[i:i + seq_len])

    return x, y


def split_windows(x, y):

    train_idx = np.where(y == 0)[0]
    test_idx = np.arange(len(y))

    np.random.shuffle(train_idx)

    return train_idx, test_idx