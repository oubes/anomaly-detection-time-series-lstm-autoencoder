# ---- IMPORTS ---- #
import torch
import os
from pathlib import Path

from utils.seed import set_seed
from data.loader import load_series, load_windows, build_labels
from experiments.search import search


# ---- PROJECT ROOT ---- #
BASE_DIR = Path(__file__).resolve().parent.parent


# ---- CONFIG ---- #
DATASET_PATH = BASE_DIR / "assets" / "data" / "art_daily_flatmiddle.csv"
DATASET_ID = "artificialWithAnomaly/art_daily_flatmiddle.csv"
WINDOWS_PATH = BASE_DIR / "assets" / "labels" / "combined_windows.json"


# ---- MAIN ENTRY ---- #
if __name__ == "__main__":

    # ---- DEVICE SETUP ---- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # ---- DEBUG PATHS (remove later if not needed) ---- #
    print("CWD:", os.getcwd())
    print("DATASET_PATH:", DATASET_PATH)
    print("WINDOWS_PATH:", WINDOWS_PATH)
    print("DATASET EXISTS:", DATASET_PATH.exists())

    # ---- HARD FAIL IF PATH WRONG ---- #
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Windows file not found: {WINDOWS_PATH}")

    # ---- REPRODUCIBILITY ---- #
    set_seed(42)

    # ---- DATA LOADING ---- #
    df = load_series(str(DATASET_PATH))

    windows = load_windows(
        str(WINDOWS_PATH),
        DATASET_ID
    )

    df = build_labels(df, windows)

    # ---- EXPERIMENT RUN ---- #
    results = search(df, device)

    print(results)