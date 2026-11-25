import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch


def set_seed(seed):
    if seed is None:
        return
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed):
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)
