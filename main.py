from runs.kend import run_all_experiments
import os

if __name__ == '__main__':
    start_from = 0
    output_dir = './depletion/v7'
    os.makedirs(output_dir, exist_ok=True)

    run_all_experiments(start_from=start_from, output_dir=output_dir, skip_prob=0.7)