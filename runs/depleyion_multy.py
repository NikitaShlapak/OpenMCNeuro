import random
import time

import numpy as np
from tqdm import tqdm

from config import GeometryConfig, MaterialConfig
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
        },
        'power': 30_000
    }
    params_grid = ParamGrid(
        {
            'tvel_r': list(np.arange(0.7, 1.5, 0.2)),
            'tvel_dist': list(np.arange(1.0, 3.0, 0.5)),
            'cladding_thick': [0.2],
            'fuel_enr': list(np.arange(3, 5, 0.5)),
            'fuel_density': list(np.arange(10, 20, 5)),
            'coolant_density': list(np.arange(0.5, 1.5, 0.5)),
            'cladding_density': [3.],
            'power': list(np.arange(1_000, 100_000, 10_000))
        }
    )

    for param_dict in params_grid.grid:
        if random.random() < prob:
            continue
        yield update_params(params, param_dict)


def run_one_experiment(
        params: dict[str, dict[str]],
        output_dir: str = 'results/depletion/'
):
    params['start_time'] = time.strftime("%Y-%m-%d_%H:%M:%S")
    return run_experiment(GeometryConfig(), MaterialConfig(), params, output_dir)

def main(start_from=0, output_dir='results/depletion/', skip_prob =0.5):
    for i, p in tqdm(enumerate(iter_param_grid(skip_prob))):
        if i < start_from:
            continue
        else:
            p['experiment_number'] = i
            try:
                run_one_experiment(p, output_dir)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    start_from = 0
    output_dir = 'results/depletion/v4'
    main(start_from=start_from, output_dir=output_dir, skip_prob=0.5)
