import numpy as np
import math

def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten()
    rmse = math.sqrt(np.mean(diff ** 2.))
    if rmse == 0:
        return float('inf')
    return 20 * math.log10(target.max() / rmse)

def angle_range(N_a):
    return np.linspace(0, np.pi, N_a, endpoint=False)