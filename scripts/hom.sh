#!/bin/bash
#SBATCH --job-name=hom
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/hom_%j.out
#SBATCH --error=logs/hom_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
module load cuda/11.6.2
conda activate trihom
export DGLBACKEND=pytorch
cd ~/Disentangle_GraphHom

for DS in cora citeseer pubmed amazon-photo amazon-computer coauthor-cs wikics \
          roman-empire amazon-ratings minesweeper tolokers questions \
          squirrel-filtered chameleon-filtered actor texas-4-classes cornell wisconsin; do
  for H in node_homo edge_homo class_homo adj_homo den_homo two_hop_homo neibh_homo \
           label_info agg_homo_soft h_s \
           localsim_cos_homo localsim_euc_homo attr_homo cls_ctrl_feat_homo h_f; do
    echo "=== $DS / $H ==="
    python homophily_test.py --dataset $DS --homophily_metric $H
  done
done
