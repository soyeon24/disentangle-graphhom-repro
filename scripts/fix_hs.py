from datasets import Dataset
import pandas as pd

DS = ['cora','citeseer','pubmed','amazon-photo','amazon-computer','coauthor-cs','wikics',
      'roman-empire','amazon-ratings','minesweeper','tolokers','questions',
      'squirrel-filtered','chameleon-filtered','actor','texas-4-classes','cornell','wisconsin']

rows = []
for name in DS:
    try:
        d = Dataset(name=name, model_name='GCN', add_self_loops=False, device='cpu')
        _, h_N = d.structural_homophily
        rows.append({'dataset': name, 'h_s_fixed': float(h_N.mean())})
        print(name, round(float(h_N.mean()), 4), flush=True)
    except Exception as e:
        print(name, 'FAILED', type(e).__name__, e, flush=True)

pd.DataFrame(rows).set_index('dataset').to_csv('experiments/h_s_fixed.csv')
