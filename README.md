# FSEML Demo

This folder contains a minimal demo package for running FSEML evaluation on Omniglot.

## Environment

Use the project Python environment:

```bash
conda activate pytorch
```

Or call Python directly:

```bash
/opt/anaconda3/envs/pytorch/bin/python
```

Install dependencies if needed:

```bash
/opt/anaconda3/envs/pytorch/bin/python -m pip install -r requirements_new2026.txt
```

## Quick Evaluation Demo

From this `FSEML-demo` folder:

```bash
cd FSEML-master
/opt/anaconda3/envs/pytorch/bin/python evaluate_classification.py --runs 1 --schedule 10 --epoch 1 --num-workers 0
```

The default model path used by `evaluate_classification.py` is:

```text
../PreNet/FSEML_Model_20260422_lat128_from93.net
```

The included dataset path matches the script's default:

```text
../data/omni/omniglot-py
```

## Experiment Configurations

The `configs/` directory provides the hyperparameters and reproduction settings. Parameters stated in the manuscript are used as defaults; parameters not specified in the manuscript use the implementation defaults in this demo.

- `configs/omniglot_eval_demo.json`: exact quick demo command and settings.
- `configs/omniglot_eval_full_reference.json`: default Omniglot evaluation schedule exposed by `evaluate_classification.py`.
- `configs/fseml_er_training_reference.json`: reference MAML, FSSAE, and replay settings exposed by `mrcl_classification.py`, including learning rates, sparsity coefficient, reconstruction weight, buffer size, replay gap, replay rate, and checkpoint settings.

The main manuscript-aligned settings are:

- Omniglot uses 1,623 character classes: the first 963 classes for meta-training and the remaining 660 classes for meta-testing.
- Each Omniglot class has 20 examples; 15 samples are used for support/fine-tuning and the remaining 5 samples are used for evaluation.
- CIFAR100 protocols group every 5, 10, or 20 classes into one class batch, with 60% of class batches used for training and the remainder for testing.
- ImageNet1000 is organized into 20 tasks, each containing 50 classes, with 10 meta-training tasks and 10 meta-test tasks.
- FSEML-ER uses replay buffer size `M = 1000`, replay gap `GI = 960`, replay rate `r = 5%`, 20,000 meta-updates, meta-update learning rate `0.0005`, inner-loop learning rate `0.1`, and weight decay `0.1`.
- Reported evaluation uses 5 independent runs and reports average and standard deviation.

## Included

- Source code needed for training/evaluation entry points.
- Pretrained FSEML model used by the evaluation script.
- Omniglot evaluation split for the demo.
- Experiment configuration files and hyperparameter settings.
- Dependency file.

## Excluded

- Git history and IDE files.
- Python caches.
- Old experiment results and logs.
- Replay visualization images.
- Extra model checkpoints not used by this demo.
