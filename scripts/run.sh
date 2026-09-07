#!/bin/bash
#SBATCH --job-name=trihom
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
module load cuda/11.6.2
conda activate trihom
export DGLBACKEND=pytorch
cd ~/Disentangle_GraphHom

for DS in cora citeseer pubmed amazon-photo amazon-computer coauthor-cs wikics \
          roman-empire amazon-ratings minesweeper tolokers questions \
          squirrel-filtered chameleon-filtered actor texas-4-classes cornell wisconsin; do
  for M in ResNet GCN SAGE GAT; do
    echo "=== $DS / $M ==="
    python train.py --dataset $DS --model $M --num_layers 2 --device cuda:0 \
      --save_dir experiments/result.csv --num_steps 1000 --hidden_dim 256 \
      --dropout 0.4 --lr 0.001 --num_runs 10
  done
done
