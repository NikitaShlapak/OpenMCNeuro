from runs.depleyion_multy import main

if __name__ == '__main__':
    start_from = 15
    output_dir = 'results/depletion/v6'
    main(start_from=start_from, output_dir=output_dir, skip_prob=0.7)