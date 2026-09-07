import re, pandas as pd, glob

log = sorted(glob.glob('logs/hom_*.out'))[-1]
rows, cur = [], None

for line in open(log):
    m = re.match(r'=== (\S+) / (\S+) ===', line)
    if m:
        cur = (m.group(1), m.group(2))
    elif 'Homophily level' in line and cur:
        v = re.search(r'is (-?[\d.eE+-]+) using', line)
        if v:
            rows.append({'dataset': cur[0], 'metric': cur[1], 'value': float(v.group(1))})
        cur = None

df = pd.DataFrame(rows)
p = df.pivot(index='dataset', columns='metric', values='value')
p.to_csv('experiments/homophily.csv')
print(p.round(4).to_string())
print(f'\n{len(df)} values collected from {log}')
