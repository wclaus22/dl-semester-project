# ETHZ Deep Learning Course Research Project

This repository contains all the code submitted with the report for the Deep Learning HS19 course. This work was a collaboration of multiple team members.

**Requirements:**
Install Python packages in your virtualenv with `pip install -r requirements.txt`

**Training Sparse Resnets:**
How to train ResNet-50 on CIFAR 100 with sparse methods:

Call `train.py --help` for more information on the argparser.

If training on the Leonhard cluster set: `--leonhard=True`

**Training Types:**

`train.py --training_type='Vanilla'` Vanilla Baseline ResNet-50

`train.py --training_type='SDR'` SDR Baseline ResNet-50

`train.py --training_type='RigL' --fraction='0.01'` RigL at 99% sparsity

`train.py --training_type='SNFS' --fraction='0.01'` SNFS at 99% sparsity

`train.py --training_type='sigma-redistribution' --fraction='0.01'` sigma-redistribtion at 99% sparsity

`train.py --training_type='sigma-pruning-rigl' --fraction='0.01'` sigma-pruning with RigL at 99% sparsity

`train.py --training_type='sigma-pruning-SNFS' --fraction='0.01'` sigma-pruning with SNFS at 99% sparsity

Default hyperparameters for ResNet-50 on CIFAR 100 are listed in Table 1 in the report.
