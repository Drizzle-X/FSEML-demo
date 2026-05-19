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


## Included

- Source code needed for training/evaluation entry points.
- Pretrained FSEML model used by the evaluation script.
- Omniglot evaluation split for the demo.
- Experiment configuration files and hyperparameter settings.
- Dependency file.

