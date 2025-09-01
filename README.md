#  Spotlighter: Revisiting Prompt Tuning from a Representative Mining View

This repository contains the implementation of the EMNLP2025 paper: Spotlighter: Revisiting Prompt Tuning from a Representative Mining View [[Paper]](https://arxiv.org/abs/). 
 
In this work, we propose **Spotlighter**, a lightweight token-selection framework that simultaneously enhances accuracy and efficiency in prompt tuning. Spotlighter evaluates each visual token's activation from both sample-wise and semantic-wise perspectives and ***retains only the top-scoring tokens*** for downstream prediction. A class-specific semantic memory bank of learned prototypes refines this selection, ensuring semantic representativeness and compensating for discarded features. To further prioritize informative signals, we introduce a two-level ranking mechanism that dynamically weights token–prototype interactions. The whole framework is shown in the figure below.
![](/framework.png "framework")

Extensive experiments conducted across 11 benchmark datasets demonstrate the effectiveness of our proposed method. Compared to CLIP and CLIPFit, our approach achieves consistent improvements in both harmonic mean accuracy (HM) and computational speed, with an improvement of 11.19% / 3.86% in HM score and 0.8K/3.8K more FPS, respectively.
Remarkably, these gains come at the cost of only **21** additional parameters, highlighting the efficiency and scalability of our design.
![](/result.png "result") 

## How to Run

We provide the running scripts in `scripts/spotlighter`, which allow you to reproduce the results on the paper.

Make sure you change the path in `DATA` and run the commands under the main directory `Spotlighter/`.  

### Install

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n spotlighter python=3.8

# Activate the environment
conda activate spotlighter

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Clone Spotlighter code repository and install requirements
```bash
# Clone Spotlighter code base
git clone https://github.com/greatest-gourmet/Spotlighter.git


# Install requirements
cd Spotlighter/
pip install -r requirements.txt
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### Datasets

Please follow the instructions at [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) in [CoOp](https://github.com/KaiyangZhou/CoOp) to prepare all datasets.

## Generalization From Base to New Classes

You will need both `scripts/spotlighter/base2new_train.sh` and `scripts/spotlighter/base2new_test.sh`. The former trains a model on base classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `Spotlighter/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# seed=1
bash scripts/spotlighter/base2new_train.sh imagenet 1
bash scripts/spotlighter/base2new_test.sh imagenet 1
```
For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on ImageNet using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– Spotlighter/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |–– seed1/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– Spotlighter/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |–– seed1/
```


## Domain Generalization

The relevant scripts are `scripts/spotlighter/xd_train.sh` and `scripts/spotlighter/xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run
```bash
# seed=1
bash scripts/spotlighter/xd_train.sh 1
```

Then, you evaluate the model on other datasets, e.g.,

```bash
bash scripts/spotlighter/xd_test.sh imagenetv2 1
 
```
##
We also provide codes for Spotlighter plugging in PromtKD in tkmc.
