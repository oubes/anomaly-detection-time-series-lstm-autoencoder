# ---- IMPORTS ---- #
import itertools
import pandas as pd

from experiments.experiment import run_experiment
from config.config import CONFIG


# ---- GRID SEARCH ---- #
def search(df, device):

    # ---- SETUP ---- #
    keys = list(CONFIG["search_space"].keys())  # type: ignore
    combos = list(itertools.product(*CONFIG["search_space"].values()))  # type: ignore

    results = []

    # ---- EXECUTION LOOP ---- #
    for i, vals in enumerate(combos):

        params = dict(zip(keys, vals))

        print(f"\nRUN {i+1}/{len(combos)}")

        res = run_experiment(df, params, device)
        results.append(res)

    return pd.DataFrame(results)