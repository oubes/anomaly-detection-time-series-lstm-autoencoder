# ---- IMPORTS ---- #
import json
import pandas as pd
import numpy as np


# ---- LOAD SERIES ---- #
def load_series(path):

    # ---- LOAD DATA ---- #
    df = pd.read_csv(path)

    # ---- TIMESTAMP ---- #
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- NORMALIZE NUMERIC FEATURES ---- #
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

    return df


# ---- LOAD WINDOWS ---- #
def load_windows(path, key):

    # ---- READ FILE ---- #
    with open(path, "r") as f:
        data = json.load(f)

    return data[key]


# ---- BUILD LABELS ---- #
def build_labels(df, windows):

    # ---- INIT LABELS ---- #
    y = np.zeros(len(df))

    # ---- MAP WINDOWS ---- #
    for s, e in windows:

        # ---- PARSE TIME ---- #
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)

        # ---- CREATE MASK ---- #
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)

        # ---- ASSIGN LABELS ---- #
        y[mask] = 1

    df["label"] = y

    return df