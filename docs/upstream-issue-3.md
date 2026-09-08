# Archived copy — upstream issue #3

Filed against [zylMozart/Disentangle_GraphHom](https://github.com/zylMozart/Disentangle_GraphHom)
at commit [`fea91b7`](https://github.com/zylMozart/Disentangle_GraphHom/commit/fea91b7735ec52f3e747f1fdd3d61c79db3c03f6):
**[issue #3](https://github.com/zylMozart/Disentangle_GraphHom/issues/3)**, opened 2026-09-08.

Title as submitted: `Released code does not reproduce Table 3 (structural homophily)`

The body below is reproduced verbatim from the submitted issue, so the record survives if
the upstream issue is later edited or removed.

---

## Summary

When I run the released code, the structural homophily value it reports does not match Table 3 of the paper. The `structural_homophily` property returns two values, and the call site uses the first one, which is not the quantity given by Definition 2. For Cora the released code gives **0.1191** where Table 3 reports **0.6164**. Taking the second return value instead reproduces Table 3 on all 18 datasets I ran.

## Environment

- Repository at commit [`fea91b7`](https://github.com/zylMozart/Disentangle_GraphHom/commit/fea91b7735ec52f3e747f1fdd3d61c79db3c03f6) (branch `master`)
- Python 3.7.12, PyTorch 1.12.0+cu116, PyG 2.3.1, DGL 1.1.2, CUDA 11.6.2
- Single NVIDIA RTX 3090

Command used:

```bash
python homophily_test.py --dataset cora --homophily_metric h_s
```

## Observed values

| Dataset | Released code (first return value) | Second return value, `h_N.mean()` |
|---|---|---|
| cora | 0.1191 | 0.6164 |
| citeseer | 0.1938 | 0.3909 |
| pubmed | 0.2962 | 0.3792 |
| amazon-photo | 0.1224 | 0.7559 |
| amazon-computer | 0.0694 | 0.7628 |
| coauthor-cs | 0.0532 | 0.7213 |
| wikics | 0.0923 | 0.6366 |
| roman-empire | 0.1207 | 0.5271 |
| amazon-ratings | 0.2122 | 0.5256 |
| minesweeper | 0.1499 | 0.7070 |
| tolokers | 0.2194 | 0.5451 |
| questions | 0.2257 | 0.5576 |
| squirrel-filtered | 0.1379 | 0.6030 |
| chameleon-filtered | 0.1788 | 0.5367 |
| actor | 0.2457 | 0.3841 |
| texas-4-classes | 0.1794 | 0.5158 |
| cornell | 0.2878 | 0.3676 |
| wisconsin | 0.2036 | 0.4687 |

The right-hand column agrees with Table 3 to four decimal places on all 18 datasets. The left-hand column is what the released code prints.

## Where the difference comes from

In `datasets.py` (L262-L289), `structural_homophily` computes the per-class terms of Definition 2 inside the loop and collects them in `h_N`:

```python
for c in range(self.num_class):
    c_dist = dist[labels==c]
    if c_dist.shape[0]==1:
        continue
    else:
        std_list = c_dist.std(dim=0)
        std_max = get_max_std(c_dist.shape[1])
        h_N_item = (1-std_list/std_max).mean()
        h_N.append(h_N_item)
h_N = torch.stack(h_N)
...
return std_list.mean(), h_N
```

`std_list` is reassigned on every iteration, so after the loop it holds the per-component standard deviations of the **final class only**. The first return value, `std_list.mean()`, is therefore a raw standard deviation for a single class — the `1 - sigma/sigma_max` normalisation of Definition 2 is never applied to it. The second return value, `h_N`, is the stack of per-class `(1 - std_list/std_max).mean()` terms, i.e. the quantity Definition 2 defines.

The call site in `homophily_test.py` (L113) takes the first value:

```python
elif homophily_metric=='h_s':
    h_N,h_N_lst = dataset.structural_homophily
    homophily_lvl = h_N
```

The names at the call site suggest an aggregate value followed by a per-class list. The second position still delivers that list; the first now delivers `std_list.mean()`. The commented-out line just below the active `return` shows an earlier three-value form whose first two positions match those names:

```python
# return h_N_cls_weighted, h_N, std_list.mean()
```

That is `(aggregate, per-class list, extra)`. In the current two-value form the aggregate is gone and `std_list.mean()` has taken the first position, while the call site still reads it as the aggregate.

Permalinks:
- `datasets.py`, the property: https://github.com/zylMozart/Disentangle_GraphHom/blob/fea91b7735ec52f3e747f1fdd3d61c79db3c03f6/datasets.py#L262-L289
- `datasets.py`, the return statement: https://github.com/zylMozart/Disentangle_GraphHom/blob/fea91b7735ec52f3e747f1fdd3d61c79db3c03f6/datasets.py#L288
- `homophily_test.py`, the call site: https://github.com/zylMozart/Disentangle_GraphHom/blob/fea91b7735ec52f3e747f1fdd3d61c79db3c03f6/homophily_test.py#L113

The two numbers are related exactly. `std_max` is a scalar, so `(1 - std_list/std_max).mean() == 1 - std_list.mean()/std_max`. For Cora (C = 7, `sigma_max = sqrt((1-1/7)/7) = 0.34993`):

```
1 - 0.1191 / 0.34993 = 0.6597
```

which is the per-class value of the last class processed, consistent with `std_list` surviving the loop.

## After using the second return value

Averaging `h_N` reproduces Table 3 on all 18 datasets I ran. Cora: 0.6164.

<details>
<summary>Self-contained check for Cora (numpy/scipy only, no repo dependencies)</summary>

```python
import pickle, numpy as np, scipy.sparse as sp
# ind.cora.* from https://github.com/kimiyoung/planetoid/tree/master/data
load = lambda n: pickle.load(open(f"ind.cora.{n}", "rb"), encoding="latin1")
ally, ty, graph = load("ally"), load("ty"), load("graph")
ridx = np.loadtxt("ind.cora.test.index", dtype=int)
oh = np.vstack((ally, ty)); oh[ridx, :] = oh[np.sort(ridx), :]
labels = oh.argmax(1); N, C = oh.shape

r, c_ = zip(*((u, v) for u, ns in graph.items() for v in ns))
A = sp.coo_matrix((np.ones(len(r)), (r, c_)), shape=(N, N)).tocsr()
A = A.maximum(A.T); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1.0

dist = A @ oh.astype(float)
dist = dist / dist.sum(1, keepdims=True)
std_max = np.sqrt((1 - 1/C) / C)

h_N, std_list = [], None
for c in range(C):
    cd = dist[labels == c]
    if cd.shape[0] == 1: continue
    std_list = cd.std(axis=0, ddof=1)          # torch .std() is unbiased by default
    h_N.append((1 - std_list / std_max).mean())

print("first  return value:", std_list.mean())   # 0.119071
print("second return value:", np.mean(h_N))      # 0.616360  == Table 3
```
</details>
