import torch
from utils.seed import set_seed
from data.loader import load_series, load_windows, build_labels
from experiments.search import search


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("DEVICE:", device)

    set_seed(42)

    df = load_series("assets/data/art_daily_flatmiddle.csv")

    windows = load_windows(
        "assets/labels/combined_windows.json",
        "artificialWithAnomaly/art_daily_flatmiddle.csv"
    )

    df = build_labels(df, windows)

    results = search(df, device)

    print(results)