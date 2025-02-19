from runs.depleyion_multy import main

if __name__ == '__main__':
    start_from = 0
    output_dir = 'results/depletion/v4'
    main(start_from=start_from, output_dir=output_dir, skip_prob=0.5)