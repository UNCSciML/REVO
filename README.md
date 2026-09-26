# REVO

Code for REVO. The implementation is built on
[verl](https://github.com/volcengine/verl) (Apache-2.0), vendored under `verl/`.

## Setup

Python 3.10, CUDA 12, 4 GPUs (tested on 80 GB A100s).

```bash
conda create -n revo python=3.10 -y
conda activate revo
pip install -r requirements.txt
```

## Training

We provide the training script below:

```bash
bash run_qwen3_1.7b_base.sh
```

By default the script uses GPUs `0,1,2,3` and downloads `Qwen/Qwen3-1.7B-Base`
and `Qwen/Qwen3-4B` from the Hugging Face Hub. You can override these with
environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
STUDENT_MODEL=/path/to/Qwen3-1.7B-Base \
TEACHER_MODEL=/path/to/Qwen3-4B \
bash run_qwen3_1.7b_base.sh
```

Training runs for 50 steps and saves a checkpoint every 10 steps under
`checkpoints/`. Extra Hydra overrides can be appended to the command line.
Logging goes to the console and Weights & Biases; W&B runs offline unless
`WANDB_API_KEY` is set.

## Data

- `datasets/dapo-math-17k-ttrl.parquet`: training prompts (DAPO-Math-17k).
- `datasets/test_data/{AIME24,AIME25,AMC23}/test.parquet`: evaluation sets.
