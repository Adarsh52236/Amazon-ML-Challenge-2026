import pandas as pd
import numpy as np

df = pd.read_csv('output/phase15_fns.tsv', sep='\t')

missing_t_addr = df['target_address'].isna() | (df['target_address'] == '<NULL>')
missing_s_addr = df['business_address'].isna() | (df['business_address'] == '<NULL>')
addr_missing = missing_t_addr | missing_s_addr

name_sim = df['name_levenshtein_sim']
weak_name = (name_sim < 0.6) & ~addr_missing

print('\n--- TYPOS / TRANSLITERATION (Weak Name, Addr Present) ---')
print(f'Count: {weak_name.sum()} ({weak_name.sum()/len(df):.1%})')
print(f'Median prob: {df[weak_name]["prob"].median():.3f}')

for i in range(min(15, weak_name.sum())):
    row = df[weak_name].iloc[i]
    print(f'S1: {row.business_name} | {row.business_address}')
    print(f'T:  {row.target_name} | {row.target_address}')
    print(f'Prob: {row.prob:.3f}')
    print('---')
