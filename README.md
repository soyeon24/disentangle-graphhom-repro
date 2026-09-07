cd ~/repro && cat > README.md << 'MDEOF'
# Disentangle_GraphHom 재현 (NeurIPS 2024)

논문 **"What Is Missing In Homophily? Disentangling Graph Homophily For Graph Neural Networks"** (NeurIPS 2024) 의 공개 코드를 서울시립대 UBAI 슈퍼컴퓨터에서 재현한 기록입니다.

- 논문: [NeurIPS 2024 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/7e810b2c75d69be186cadd2fe3febeab-Paper-Conference.pdf)
- 원본 코드: [zylMozart/Disentangle_GraphHom](https://github.com/zylMozart/Disentangle_GraphHom)

---

## 요약

| 항목 | 내용 |
|---|---|
| 학습 | 18개 데이터셋 × 4개 모델 × 10 runs = **720회** (약 25분) |
| 지표 계산 | 15종 호모필리 지표 × 18개 데이터셋 = **270회** (약 2시간) |
| 정확도 재현 | 논문 Table 4 대비 **대부분 0.5%p 이내** 일치 |
| 지표 재현 | 논문 Table 3과 **소수점 4자리까지** 일치 |

---

## 실행 환경

| 구분 | 사양 |
|---|---|
| 클러스터 | UBAI (서울시립대 도시과학빅데이터AI연구원), Slurm |
| 파티션 / GPU | `gpu1` / NVIDIA RTX 3090 |
| CUDA | `cuda/11.6.2` (module load) |
| Python | 3.7.12 (Miniconda, conda-forge) |
| PyTorch | 1.12.0+cu116 |
| PyG | torch-geometric 2.3.1 |
| DGL | 1.1.2+cu116 |

### 설치

```bash
conda create -n trihom python=3.7 -c conda-forge --override-channels -y
conda activate trihom
module load cuda/11.6.2

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# PyG 계열 - 버전 고정 필수
pip install --no-cache-dir \
  torch-scatter==2.0.9 torch-sparse==0.6.14 \
  torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f "https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html"
pip install torch-geometric==2.3.1

pip install ogb==1.3.6 networkx==2.3 packaging pyyaml tabulate matplotlib seaborn
pip install dgl==1.1.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
export DGLBACKEND=pytorch
```

### 설치 시 함정

**PyG wheel 버전을 반드시 고정해야 합니다.** 버전을 명시하지 않으면 pip가 PyPI 최신 버전(torch-scatter 2.1.1, torch-sparse 0.6.17)을 찾는데, 이 버전들은 torch-1.12.0+cu116 wheel 인덱스에 존재하지 않습니다. 결과적으로 pip가 조용히 소스 배포판(.tar.gz)으로 폴백해 CUDA 컴파일을 시도하고, 30분 이상 걸리다 실패합니다.

인덱스에 실제로 존재하는 버전은 다음으로 확인할 수 있습니다.

```bash
curl -s "https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html" \
  | grep -o '[a-z_]*-[0-9][^"<]*cp37-cp37m-linux[^"<]*' | sort -u
```

로그에 `Downloading ...whl`이 보이면 정상, `Building wheel for ... (setup.py)`가 보이면 잘못된 경로입니다.

---

## 데이터 준비

```bash
mkdir -p data experiments logs

python -c "
from preprocess_dataset import load_new_dataset
for d in ['cora','citeseer','pubmed','amazon-photo','amazon-computer','coauthor-cs','wikics']:
    load_new_dataset(dataset_name=d, split_type='random',
                     train_prop=0.6, valid_prop=0.2, num_data_splits=10)
"

git clone https://github.com/yandex-research/heterophilous-graphs.git
cp heterophilous-graphs/data/*.npz data/
```

코드는 `data/{name}.npz` 형식을 기대하며, 데이터셋 이름의 하이픈은 언더스코어로 변환됩니다 (`texas-4-classes` → `texas_4_classes.npz`).

---

## 실행

```bash
sbatch scripts/run.sh      # 학습
sbatch scripts/hom.sh      # 호모필리 지표
python scripts/collect.py  # 로그 -> results/homophily.csv
python scripts/fix_hs.py   # 구조 호모필리 재계산
python scripts/report.py   # 정확도 표 + 논문 대조
python scripts/analyze.py  # 지표 <-> 성능 상관계수
```

`--save_dir`는 디렉토리가 아니라 **CSV 파일 경로**를 받습니다. 코드가 `df.to_csv(self.save_dir, mode='a')`로 직접 쓰기 때문에, 디렉토리를 넘기면 학습을 모두 마친 뒤 저장 단계에서 `IsADirectoryError`가 발생합니다.

---

## 결과

### 노드 분류 정확도 (%)

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

각 값은 10회 실행(서로 다른 data split)의 평균입니다. 표준편차를 포함한 원본은 `results/result.csv`에 있습니다.

### 논문 Table 4 대조 (GCN)

| Dataset | 논문 | 재현 | 차이 |
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

11개 중 8개가 0.5%p 이내입니다. 전반적으로 재현값이 근소하게 낮은 경향이 있는데, 논문의 하이퍼파라미터 그리드서치(72개 조합)를 생략하고 단일 설정(hidden_dim=256, dropout=0.4, lr=0.001, num_layers=2)으로 고정한 결과로 보입니다.

### 관찰

이종성 그래프에서 논문이 지적한 현상이 그대로 나타났습니다.

- **minesweeper**: MLP 50.72% -> GCN 89.70%. label homophily가 중간 수준(0.68)임에도 GNN 이득이 39%p에 달합니다. 라벨 기반 지표만으로는 설명되지 않는 사례입니다.
- **cornell / wisconsin**: MLP가 GCN을 각각 10.8%p, 3.7%p 앞섭니다.
- **actor**: 네 모델이 34~35%대에 밀집. 그래프 구조가 분류에 기여하지 못합니다.

### 호모필리 지표

15종 전체는 `results/homophily.csv`에 있습니다. 논문 Table 3과 대조한 일부입니다.

| Dataset | h_class (논문) | h_class (재현) | h_node (논문) | h_node (재현) |
|---|---|---|---|---|
| cora | 0.7657 | 0.7657 | 0.8252 | 0.8252 |
| roman-empire | 0.0208 | 0.0208 | 0.0460 | 0.0460 |
| amazon-ratings | 0.1266 | 0.1266 | 0.3757 | 0.3757 |

### 구조 호모필리 (h_S) 재계산

`homophily_test.py`를 그대로 실행하면 논문 Table 3과 다른 값이 나옵니다 (Cora 기준 0.1191 vs 논문 0.6164). `datasets.py`의 `structural_homophily` property가 두 개의 값을 반환하는데, 논문 Definition 2에 해당하는 것은 두 번째(`h_N`)입니다.

```python
h_N_item = (1 - std_list/std_max).mean()   # 논문 정의
h_N.append(h_N_item)
...
return std_list.mean(), h_N                # 호출부는 첫 번째를 사용
```

`h_N.mean()`을 직접 호출하면 논문 값과 일치합니다. `scripts/fix_hs.py`가 이 계산을 수행합니다.

| Dataset | 논문 Table 3 | 재계산 |
|---|---|---|
| cora | 0.6164 | 0.6164 |
| citeseer | 0.3909 | 0.3909 |
| pubmed | 0.3792 | 0.3792 |
| minesweeper | 0.7070 | 0.7070 |
| roman-empire | 0.5271 | 0.5271 |
| actor | 0.3841 | 0.3841 |
| cornell | 0.3676 | 0.3676 |

18개 데이터셋 전부 소수점 4자리까지 일치합니다.

---

## 재현하지 않은 부분

- **데이터셋 13개**: flickr, ogbn-arxiv, genius, twitch-* 시리즈. 논문은 31개를 사용하며 본 재현은 18개입니다.
- **하이퍼파라미터 그리드서치**: 논문은 데이터셋/모델당 72개 조합을 탐색합니다. 본 재현은 단일 설정을 사용했습니다.
- **합성 데이터 스윕 (Figure 2)**: h_L × h_S × h_F 격자 탐색. 약 22,000 runs 규모로 별도 계산 예산이 필요합니다.

---

## 디렉토리 구조
