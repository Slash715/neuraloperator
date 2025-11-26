#!/bin/bash -l
#SBATCH --job-name=fno_debug
#SBATCH --output=/home/jw3275/neuraloperator/scripts/logs/fno_debug_%j.out
#SBATCH --error=/home/jw3275/neuraloperator/scripts/logs/fno_debug_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00

echo "===== [DEBUG] Starting ====="
echo "Hostname: $(hostname)"
echo "Shell: $SHELL"

echo ">>> Loading miniconda"
module load miniconda  # <-- IMPORTANT

echo ">>> Activating environment"
source activate fno || conda activate fno

echo ">>> Python path:"
which python

echo ">>> Python version:"
python -V

echo ">>> Import test:"
python - << 'EOF'
import torch, yaml
print("Torch OK:", torch.__version__)
print("YAML OK:", yaml.__version__)
try:
    import neuralop
    print("NeuralOp OK")
except Exception as e:
    print("NeuralOp FAIL:", e)
EOF

echo ">>> Now running training script:"
python train_fno_inviscid.py --config config.yaml

echo "===== [DEBUG] FINISHED ====="
