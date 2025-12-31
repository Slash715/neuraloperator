#!/bin/bash 
#SBATCH --job-name=pino_burgers_100_100_burgers_v_0001
#SBATCH --output=/home/jw3275/neuraloperator/scripts/logs/pino_burgers_%j.txt
#SBATCH --error=/home/jw3275/neuraloperator/scripts/logs/pino_burgers_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jack.wang.jw3275@yale.edu

# Load environment
module load miniconda
conda activate fno     # same env you used for train_burgers_pino.py

# Move to your project folder
cd /home/jw3275/neuraloperator/scripts

# Ensure logs directory exists
mkdir -p logs

echo "===== Starting PINO Burgers Training ====="
echo "Running on:"
nvidia-smi
echo "Current directory: $PWD"

# Run training (zencfg reads config from config.burgers_pino_config.Default)
python train_burgers_v_0001_pino_res_100_modified_ite_10000.py

echo "===== Training Completed ====="