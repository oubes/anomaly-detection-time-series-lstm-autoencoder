# ---- IMPORTS ---- #
import random
import numpy as np
import torch


# ---- SEED SETTING ---- #
def set_seed(seed=42):

    # ---- PYTHON RNG ---- #
    random.seed(seed)

    # ---- NUMPY RNG ---- #
    np.random.seed(seed)

    # ---- TORCH RNG ---- #
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)