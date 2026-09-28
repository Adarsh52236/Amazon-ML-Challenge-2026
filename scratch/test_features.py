import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.features.similarity_metrics import levenshtein_similarity as ratio
import numpy as np
import re

df = pd.read_csv('output/phase15_fns.tsv', sep='\t')
weak_name = (df['name_levenshtein_sim'] < 0.6) & ~(df['target_address'].isna() | (df['target_address'] == '<NULL>'))
df_weak = df[weak_name]

def consonant_skeleton(text):
    if not isinstance(text, str): return ''
    vowels = set('aeiouyAEIOUY .,-_')
    return ''.join(c for c in text if c not in vowels).lower()

vowel_sims = []
for i in range(len(df_weak)):
    row = df_weak.iloc[i]
    c1 = consonant_skeleton(row.business_name)
    c2 = consonant_skeleton(row.target_name)
    v_sim = ratio(c1, c2)
    vowel_sims.append(v_sim)

print(f'Weak Name FNs: {len(df_weak)}')
print(f'Median standard name_sim: {df_weak["name_levenshtein_sim"].median():.3f}')
print(f'Median consonant sim: {np.median(vowel_sims):.3f}')
print(f'Consonant sim > 0.8: {(np.array(vowel_sims) > 0.8).sum()}')

# Also check token reordering similarity (e.g. token sort ratio)
def token_sort_ratio(t1, t2):
    if not isinstance(t1, str): t1 = ''
    if not isinstance(t2, str): t2 = ''
    s1 = ' '.join(sorted(re.findall(r'\w+', t1.lower())))
    s2 = ' '.join(sorted(re.findall(r'\w+', t2.lower())))
    return ratio(s1, s2)

sort_sims = []
for i in range(len(df_weak)):
    row = df_weak.iloc[i]
    sort_sims.append(token_sort_ratio(row.business_name, row.target_name))

print(f'Median sorted token sim: {np.median(sort_sims):.3f}')
print(f'Sorted token sim > 0.8: {(np.array(sort_sims) > 0.8).sum()}')
