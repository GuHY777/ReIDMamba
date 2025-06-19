import torch
import numpy as np
import random
import os


def setup_seed(seed, deterministic=True, benchmark=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cuda.enable_math_sdp(True)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False