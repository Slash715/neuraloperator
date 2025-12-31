#!/bin/bash
#SBATCH --job-name=train_burgers
#SBATCH --output=/home/jw3275/neuraloperator/scripts/logs/train_burgers.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jack.wang.jw3275@yale.edu


module load miniconda
conda activate fno

# Move to the folder containing your .py file
cd /home/jw3275/neuraloperator/scripts

# Run your converted notebook
python train_burgers.py