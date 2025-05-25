#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=slurm/%A.out
#SBATCH --error=slurm/%A.err
#SBATCH --job-name="pi0_lib"

export WANBD__SERVICE_WAIT=300

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

export TRANSFORMERS_CACHE=/iliad/u/jvclark/.cache/hub

source /iliad/u/jvclark/miniconda3/etc/profile.d/conda.sh
source /iliad/u/jvclark/miniconda3

cd /iliad/u/jvclark/openpi/


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_full_finetune --exp-name=pi0ft --overwrite

wait

# torchrun --standalone --nnodes 1 --nproc-per-node 8  vla-scripts/finetune.py --dataset_name=fullcross --batch_size 2 &

# wait

# torchrun --standalone --nnodes 1 --nproc-per-node 8  vla-scripts/finetune.py --dataset_name=fullcross --batch_size 2 &

# wait