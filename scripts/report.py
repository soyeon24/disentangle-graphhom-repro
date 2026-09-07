import pandas as pd

hom = pd.read_csv('experiments/homophily.csv', index_col=0)
hom['h_s'] = pd.read_csv('experiments/h_s_fixed.csv', index_col=0)['h_s_fixed']
acc = pd.read_csv('experiments/result.csv').drop_duplicates(subset=['dataset','model'], keep='last')
acc = acc[acc.dataset != 'syn'].pivot(index='dataset', columns='model', values='test accuracy mean')

print("="*70)
print("  Disentangle_GraphHom (NeurIPS 2024) 재현 결과")
print("="*70)
print(f"\n[1/3] 환경: UBAI gpu1 (RTX 3090) / torch 1.12.0+cu116 / PyG 2.3.1")
print(f"[2/3] 데이터셋 {len(acc)}개 x 모델 4종 x 10 runs = {len(acc)*40} 학습")
print(f"[3/3] 호모필리 지표 {len(hom.columns)}종 x {len(hom)}개 데이터셋\n")

print("-"*70)
print("NODE CLASSIFICATION ACCURACY (%)")
print("-"*70)
print((acc*100).round(2)[['ResNet','GCN','SAGE','GAT']].to_string())

print("\n" + "-"*70)
print("논문 Table 4 대조 (GCN)")
print("-"*70)
paper = {'cora':86.36,'citeseer':76.44,'pubmed':89.12,'amazon-photo':95.16,
         'amazon-computer':91.58,'coauthor-cs':95.68,'wikics':85.20,
         'roman-empire':78.76,'minesweeper':90.16,'tolokers':84.55,'actor':35.10}
for k, v in paper.items():
    if k in acc.index:
        mine = acc.loc[k,'GCN']*100
        print(f"  {k:20s} 논문 {v:6.2f}  재현 {mine:6.2f}  (차이 {mine-v:+.2f})")

print("\n" + "-"*70)
print("METRIC vs GNN PERFORMANCE CORRELATION (Table 1)")
print("-"*70)
import subprocess
print(subprocess.run(['python','analyze.py'], capture_output=True, text=True).stdout)
