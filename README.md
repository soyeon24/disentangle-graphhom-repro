# Reproducing "Disentangling Graph Homophily for GNNs" (NeurIPS 2024)

A reproduction of **"What Is Missing In Homophily? Disentangling Graph Homophily For Graph Neural Networks"** (NeurIPS 2024), run on the UBAI supercomputing cluster at the University of Seoul.

- Paper: [NeurIPS 2024 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e810b2c75d69be186cadd2fe3febeab-Paper-Conference.pdf)
- Original code: [zylMozart/Disentangle_GraphHom](https://github.com/zylMozart/Disentangle_GraphHom)

---

## Summary

| Item | Detail |
|---|---|
| Training | 18 datasets × 4 models × 10 runs = **720 runs** (~25 min) |
| Metric computation | 15 homophily metrics × 18 datasets = **270 runs** (~2 h) |
| Accuracy | Within **0.5%p of Table 4** for most datasets |
| Homophily metrics | Match **Table 3 to 4 decimal places** |

---

## Environment

| Component | Version |
|---|---|
| Cluster | UBAI (Urban Big data and AI Institute, University of Seoul), Slurm |
| Partition / GPU | `gpu1` / NVIDIA RTX 3090 |
| CUDA | `cuda/11.6.2` (via `module load`) |
| Python | 3.7.12 (Miniconda, conda-forge) |
| PyTorch | 1.12.0+cu116 |
| PyG | torch-geometric 2.3.1 |
| DGL | 1.1.2+cu116 |

### Setup

```bash
conda create -n trihom python=3.7 -c conda-forge --override-channels -y
conda activate trihom
module load cuda/11.6.2

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# PyG extensions - pinning versions is required, see note below
pip install --no-cache-dir \
  torch-scatter==2.0.9 torch-sparse==0.6.14 \
  torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f "https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html"
pip install torch-geometric==2.3.1

pip install ogb==1.3.6 networkx==2.3 packaging pyyaml tabulate matplotlib seaborn
pip install dgl==1.1.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
export DGLBACKEND=pytorch
```

### Installation pitfall

**Pin the PyG extension versions explicitly.** Without version pins, pip resolves to the latest PyPI releases (`torch-scatter 2.1.1`, `torch-sparse 0.6.17`), which are absent from the `torch-1.12.0+cu116` wheel index. Pip then silently falls back to source distributions and attempts a CUDA compile that takes 30+ minutes before failing.

To see which versions the index actually carries:

```bash
curl -s "https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html" \
  | grep -o '[a-z_]*-[0-9][^"<]*cp37-cp37m-linux[^"<]*' | sort -u
```

`Downloading ...whl` in the pip log means the wheel was found; `Building wheel for ... (setup.py)` means it was not.

---

## Data preparation

```bash
mkdir -p data experiments logs

# Datasets fetched automatically via DGL / PyG / OGB
python -c "
from preprocess_dataset import load_new_dataset
for d in ['cora','citeseer','pubmed','amazon-photo','amazon-computer','coauthor-cs','wikics']:
    load_new_dataset(dataset_name=d, split_type='random',
                     train_prop=0.6, valid_prop=0.2, num_data_splits=10)
"

# Heterophilous datasets require a manual download
git clone https://github.com/yandex-research/heterophilous-graphs.git
cp heterophilous-graphs/data/*.npz data/
```

The code expects `data/{name}.npz`, with hyphens converted to underscores (`texas-4-classes` → `texas_4_classes.npz`). Filenames in the yandex repo already follow this convention.

---

## Running

```bash
sbatch scripts/run.sh      # training
sbatch scripts/hom.sh      # homophily metrics
python scripts/collect.py  # parse logs -> results/homophily.csv
python scripts/fix_hs.py   # recompute structural homophily
python scripts/report.py   # accuracy table + comparison against the paper
python scripts/analyze.py  # metric-performance correlations
```

Note that `--save_dir` expects a **CSV file path**, not a directory. The code calls `df.to_csv(self.save_dir, mode='a')` directly, so passing a directory raises `IsADirectoryError` after training has already finished.

---

## Results

### Node classification accuracy (%)

| Dataset | MLP | GCN | SAGE | GAT |
|---|---|---|---|---|
| actor | 35.33 | 34.14 | 35.53 | 33.97 |
| amazon-computer | 83.69 | 91.55 | 90.88 | 91.68 |
| amazon-photo | 90.44 | 95.39 | 95.43 | 95.55 |
| amazon-ratings | 45.18 | 49.73 | 52.63 | 51.66 |
| chameleon-filtered | 40.31 | 43.10 | 40.80 | 41.70 |
| citeseer | 70.78 | 75.95 | 76.42 | 76.03 |
| coauthor-cs | 94.35 | 95.61 | 95.51 | 95.51 |
| cora | 68.72 | 85.91 | 86.82 | 86.80 |
| cornell | 70.81 | 60.00 | 69.19 | 64.86 |
| minesweeper | 50.72 | 89.70 | 90.45 | 89.82 |
| pubmed | 86.79 | 88.70 | 88.95 | 88.70 |
| questions | 71.25 | 75.78 | 76.33 | 77.78 |
| roman-empire | 64.78 | 77.19 | 83.59 | 83.18 |
| squirrel-filtered | 38.28 | 39.96 | 39.04 | 38.65 |
| texas-4-classes | 71.46 | 67.11 | 79.60 | 69.56 |
| tolokers | 72.89 | 83.52 | 82.67 | 83.74 |
| wikics | 81.12 | 85.03 | 85.65 | 85.66 |
| wisconsin | 79.22 | 75.49 | 82.16 | 74.71 |

Each figure is the mean over 10 runs on different data splits. Standard deviations and the full hyperparameter record are in `results/result.csv`.

### Comparison against Table 4 (GCN)

| Dataset | Paper | Reproduced | Diff |
|---|---|---|---|
| amazon-computer | 91.58 | 91.55 | -0.03 |
| coauthor-cs | 95.68 | 95.61 | -0.07 |
| wikics | 85.20 | 85.03 | -0.17 |
| amazon-photo | 95.16 | 95.39 | +0.23 |
| pubmed | 89.12 | 88.70 | -0.42 |
| cora | 86.36 | 85.91 | -0.45 |
| minesweeper | 90.16 | 89.70 | -0.46 |
| citeseer | 76.44 | 75.95 | -0.49 |
| actor | 35.10 | 34.14 | -0.96 |
| tolokers | 84.55 | 83.52 | -1.03 |
| roman-empire | 78.76 | 77.19 | -1.57 |

Eight of eleven fall within 0.5%p. The reproduced numbers skew slightly low overall, which is consistent with skipping the paper's hyperparameter grid search (72 configurations per dataset-model pair) in favour of a single fixed setting (`hidden_dim=256, dropout=0.4, lr=0.001, num_layers=2`).

### Observations

The phenomena the paper highlights show up clearly in the heterophilous datasets.

- **minesweeper**: MLP 50.72% → GCN 89.70%. Label homophily sits at a moderate 0.68, yet the GNN gain is 39%p. Label-based metrics alone do not account for this.
- **cornell / wisconsin**: MLP beats GCN by 10.8%p and 3.7%p respectively.
- **actor**: all four models cluster around 34-35%, meaning graph structure contributes essentially nothing.

### Homophily metrics

All 15 metrics are in `results/homophily.csv`. A sample compared against Table 3:

| Dataset | h_class (paper) | h_class (repro) | h_node (paper) | h_node (repro) |
|---|---|---|---|---|
| cora | 0.7657 | 0.7657 | 0.8252 | 0.8252 |
| roman-empire | 0.0208 | 0.0208 | 0.0460 | 0.0460 |
| amazon-ratings | 0.1266 | 0.1266 | 0.3757 | 0.3757 |

### Recomputing structural homophily (h_S)

Running `homophily_test.py` as-is yields values that differ from Table 3 (0.1191 vs 0.6164 for Cora). The `structural_homophily` property in `datasets.py` returns two values, and the one matching Definition 2 in the paper is the second (`h_N`):

```python
h_N_item = (1 - std_list/std_max).mean()   # matches the paper's definition
h_N.append(h_N_item)
...
return std_list.mean(), h_N                # the caller uses the first value
```

Calling `h_N.mean()` directly reproduces the published numbers. `scripts/fix_hs.py` does this.

| Dataset | Table 3 | Recomputed |
|---|---|---|
| cora | 0.6164 | 0.6164 |
| citeseer | 0.3909 | 0.3909 |
| pubmed | 0.3792 | 0.3792 |
| minesweeper | 0.7070 | 0.7070 |
| roman-empire | 0.5271 | 0.5271 |
| actor | 0.3841 | 0.3841 |
| cornell | 0.3676 | 0.3676 |

All 18 datasets match to four decimal places.

The first return value is `std_list.mean()`, where `std_list` is left over from the final
loop iteration — a raw standard deviation for the last class processed, with the
`1 - sigma/sigma_max` normalisation never applied. Since `std_max` is a scalar,
`1 - 0.1191/0.34993 = 0.6597` is exactly that class's per-class value, which is why the
number looks plausible enough to pass unnoticed.

Reported upstream on 2026-09-08:
[zylMozart/Disentangle_GraphHom#3](https://github.com/zylMozart/Disentangle_GraphHom/issues/3).

---

## Not reproduced

- **13 datasets**: flickr, ogbn-arxiv, genius, and the twitch-* series. The paper uses 31 datasets; this reproduction covers 18.
- **Hyperparameter grid search**: the paper explores 72 configurations per dataset-model pair. A single fixed configuration was used here.
- **Synthetic sweep (Figure 2)**: the h_L × h_S × h_F grid, roughly 22,000 runs, which needs a separate compute budget.

---

## Repository layout

```
.
├── README.md
├── scripts/
│   ├── run.sh          # Slurm - training batch
│   ├── hom.sh          # Slurm - homophily metric batch
│   ├── collect.py      # log parsing -> homophily.csv
│   ├── fix_hs.py       # structural homophily recomputation
│   ├── report.py       # accuracy table + paper comparison
│   └── analyze.py      # metric-performance correlations
├── results/
│   ├── result.csv      # training results with full hyperparameters
│   ├── homophily.csv   # 15 metrics × 18 datasets
│   └── h_s_fixed.csv   # recomputed h_S
└── logs/
    ├── train.out
    └── homophily.out
```

---

## Acknowledgement

The author acknowledges the Urban Big data and AI Institute of the University of Seoul supercomputing resources (http://ubai.uos.ac.kr) made available for conducting the research reported in this repository.
