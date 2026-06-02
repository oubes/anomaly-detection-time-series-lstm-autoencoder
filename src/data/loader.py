import json
import pandas as pd
import numpy as np


def load_series(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    num_cols = df.select_dtypes(include=["number"]).columns

    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

    return df


def load_windows(path, key):
    with open(path, "r") as f:
        return json.load(f)[key]


def build_labels(df, windows):
    y = np.zeros(len(df))

    for s, e in windows:
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)

        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
        y[mask] = 1

    df["label"] = y
    return df