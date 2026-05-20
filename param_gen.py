from itertools import product
import numpy as np

def frange(start, stop, step, decimals=10):
    return np.round(np.arange(start, stop + step / 2, step), decimals)

param_ranges = {
    "H_K": [0.5],
    "M_S": [0.65],
    "alpha": [1e-3, 5e-3],
    "phi": frange(5, 20, 5),
    "g": [2.0],
    "f": [36],
}

parameter_sets = [
    dict(zip(param_ranges.keys(), values))
    for values in product(*param_ranges.values())
]

print(parameter_sets)