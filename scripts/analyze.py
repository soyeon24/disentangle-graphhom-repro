import pandas as pd, numpy as np

hom = pd.read_csv('experiments/homophily.csv', index_col=0)
acc = pd.read_csv('experiments/result.csv').drop_duplicates(
        subset=['dataset','model'], keep='last')
acc = acc[acc.dataset != 'syn'].pivot(
        index='dataset', columns='model', values='test accuracy mean')

C   = {'roman-empire':18,'amazon-ratings':5,'minesweeper':2,'tolokers':2,
       'questions':2,'squirrel-filtered':5,'chameleon-filtered':5,'actor':5,
       'texas-4-classes':5,'cornell':5,'wisconsin':5,'cora':7,'citeseer':6,
       'pubmed':3,'amazon-photo':8,'amazon-computer':10,'coauthor-cs':15,
       'wikics':10}
rho = {'roman-empire':4.58,'amazon-ratings':20.39,'minesweeper':7.99,
       'tolokers':392.36,'questions':95.31,'squirrel-filtered':206.02,
       'chameleon-filtered':78.05,'actor':37.37,'texas-4-classes':10.98,
       'cornell':10.08,'wisconsin':11.88,'cora':14.39,'citeseer':13.74,
       'pubmed':23.24,'amazon-photo':122.54,'amazon-computer':169.71,
       'coauthor-cs':24.60,'wikics':149.77}

df = hom.join(acc, how='inner')
df['C'] = pd.Series(C)
df['rho'] = pd.Series(rho)

hL, hS, hF = df['node_homo'], df['h_s'], df['h_f']
c, r = df['C'], df['rho']
a = (hL*c - 1)/(c - 1)
b = c*((1-hL)/(c-1))**2 + c*(1-hS)**2/(c-1) + a**2
w = hF/r

df['TriHom_noG'] = (1 - w**2 * b)/(1 - w*a)**2
df['TriHom_G'] = a**2/b * df['TriHom_noG']

models = ['ResNet','GCN','SAGE','GAT']
metrics = [m for m in hom.columns if m != 'attr_homo'] + ['TriHom_noG','TriHom_G']

out = pd.DataFrame({m: df[metrics].corrwith(df[m]) for m in models})
out['avg'] = out.abs().mean(axis=1)
print(out.round(4).sort_values('avg', ascending=False).to_string())
