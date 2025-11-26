#!/bin/bash
#SBATCH --job-name=fno_inviscid
#SBATCH --output=/home/jw3275/neuraloperator/scripts/logs/fno_inviscid_%j.txt
#SBATCH --error=/home/jw3275/neuraloperator/scripts/logs/fno_inviscid_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jack.wang.jw3275@yale.edu

# Load environment
module load miniconda
conda activate fno     # same env you used previously

# Move to your project folder
cd /home/jw3275/neuraloperator/scripts/fno_inviscid

echo "===== Starting FNO Inviscid Burgers Training ====="
echo "Running on:"
nvidia-smi
echo "Current directory: $PWD"

# Run training
python train_fno_inviscid.py --config config.yaml

echo "===== Training Completed ====="
