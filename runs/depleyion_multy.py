import os
import random
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from config import MaterialConfig, GeometryConfig
from runs.depletion_single import run_experiment
from utils.experiments import ParamGrid

random.seed(0)


def update_params(old_params: dict[str, dict], new_params: dict[str]) -> dict[str, dict]:
    for key, value in old_params.items():
        if isinstance(value, dict):
            update_params(value, new_params)
        elif key in new_params.keys():
            old_params[key] = new_params[key]
    return old_params


def iter_param_grid(prob: float = 1.0):
    assert 0 < prob <= 1, "Probability must be between 0 and 1"
    params = {
        'materials': {
            'fuel_enr': 5,
            'fuel_density': 8.3,
            'coolant_density': 1,
            'cladding_density': 2
        },
        'geometry': {
            'tvel_r': 1.0,
            'tvel_dist': 2.0,
            'cladding_thick': 0.1
        }
    }
    params_grid = ParamGrid(
        {
            'tvel_r': list(np.arange(0.9, 1.0, 0.1)),
            'tvel_dist': list(np.arange(2.0, 5.0, 0.5)),
            'cladding_thick': [0.1, 0.15],
            'fuel_enr': list(np.arange(3, 5, 0.5)),
            'fuel_density': list(np.arange(8, 9, 0.5)),
            'coolant_density': list(np.arange(0.8, 1.1, 0.05)),
            'cladding_density': [1, 1.5, 2]
        }
    )

    for param_dict in params_grid.grid:
        if random.random() < prob:
            continue
        yield update_params(params, param_dict)


def run_one_experiment(params: dict[str, dict[str]]):
    params['start_time'] = time.strftime("%Y-%m-%d_%H:%M:%S")
    return run_experiment(GeometryConfig(), MaterialConfig(), params)


if __name__ == '__main__':
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     executor.map(run_experiment, iter_param_grid(prob=0.5))
    for i, p in enumerate(iter_param_grid(0.5)):
        p['experiment_number'] = i
        run_one_experiment(p)
