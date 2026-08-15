import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    Copied from llava.utils
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
