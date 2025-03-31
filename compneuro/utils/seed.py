import random
import numpy as np
import torch


def seed_everything(seed: int = 0) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed