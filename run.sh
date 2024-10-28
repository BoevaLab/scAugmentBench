# Tuning Representation Dimension and Projection Dimension
echo "This is an example to find parameters for the simclr method"
python main.py --multirun +exp=simclr_ablation


# Ablation of Augmentation Hyperparameters
echo "This is an example to tune augmentation-parameters CLEAR-augmentations"
python main.py --multirun +aug_exp=combi-bbknn-all
python main.py --multirun +aug_exp=combi-gauss-all
python main.py --multirun +aug_exp=combi-mask-all
python main.py --multirun +aug_exp=combi-swap-all

# Benchmarking Unimodal Batch-Correction
python main.py --multirun +exp=all_datasets

# Unimodal Query-Reference Mapping Benchmark
python main.py --multirun +exp=qr_big
python main.py --multirun +exp=qr_small
