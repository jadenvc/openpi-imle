#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=slurm/%A.out
#SBATCH --error=slurm/%A.err
#SBATCH --job-name="eval_pi0"

echo "SLURM_JOBID=     $SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=    $SLURM_NNODES"
echo "WORKDIR=         $SLURM_SUBMIT_DIR"

# go to your repo root
cd /iliad/u/jvclark/openpi

# cache location for HuggingFace models
export TRANSFORMERS_CACHE=/iliad/u/jvclark/.cache/hub

# allow your code to import the local libero package
export PYTHONPATH="$PWD/third_party/libero:$PYTHONPATH"

#
# 1) Launch the policy‐server WITHOUT activating the Libero venv
#
# uv run scripts/serve_policy.py --env LIBERO &
uv run scripts/serve_policy.py policy:checkpoint --policy.config=imle_libero_full_finetune --policy.dir=checkpoints/imle_libero_full_finetune/imle_libero_full_parity/118000 &
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero_full_finetune --policy.dir=checkpoints/pi0_libero_full_finetune/pi0_libero_full/90000 &
SERVER_PID=$!

# give the server time to come online
# sleep 5

#
# 2) Now activate the Libero venv and run your simulation
#
source examples/libero/.venv/bin/activate
# (you can re‑export PYTHONPATH here if needed, but it inherits from above)

python examples/libero/main.py

#
# 3) Shut down the policy‐server
#
kill $SERVER_PID
wait
