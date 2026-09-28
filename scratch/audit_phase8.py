import pandas as pd

df = pd.read_csv('output/phase_8_correct_cap_ablation.tsv', sep='\t')
df_fold = pd.read_csv('output/phase_8_fold_level_results.tsv', sep='\t')

print("1. CAP 100 THRESHOLD TABLE")
c100 = df[df['Cap'] == '100'].sort_values('Threshold')
print(c100[['Threshold', 'Macro_F05_Mean', 'Precision', 'Model_Recall', 'ZM_FP_Rate', 'Singles_Acc', 'Multi_F05']].to_string(index=False))

print("\n2. CAP 100 FOLD LEVEL RESULTS AT PEAK (0.70)")
peak_thresh = 0.70
f100 = df_fold[(df_fold['Cap'] == '100') & (df_fold['Threshold'] == peak_thresh)]
print(f100[['Fold', 'Macro_F05']].to_string(index=False))
print(f"Mean: {f100['Macro_F05'].mean():.4f}, Std: {f100['Macro_F05'].std():.4f}, Min: {f100['Macro_F05'].min():.4f}, Max: {f100['Macro_F05'].max():.4f}")

print("\n3. FIXED THRESHOLD COMPARISON (0.70)")
t70 = df[df['Threshold'] == 0.70]
print(t70[['Cap', 'Macro_F05_Mean', 'ZM_FP_Rate', 'Precision', 'Model_Recall']].to_string(index=False))

print("\n4. PAIRED FOLD DIFFERENCE AT 0.70")
d100 = df_fold[(df_fold['Cap'] == '100') & (df_fold['Threshold'] == 0.70)].set_index('Fold')['Macro_F05']
for c in [125, 150, 200, 250, 300, 500]:
    dc = df_fold[(df_fold['Cap'] == str(c)) & (df_fold['Threshold'] == 0.70)].set_index('Fold')['Macro_F05']
    diff = dc - d100
    print(f"100 vs {c}: mean={diff.mean():.4f}, std={diff.std():.4f}")

print("\n5. PEAK-VS-PEAK COMPARISON")
best_100 = df[df['Cap'] == '100']['Macro_F05_Mean'].max()
for c in [125, 150, 200, 250, 300, 500]:
    best_c = df[df['Cap'] == str(c)]['Macro_F05_Mean'].max()
    print(f"100 vs {c}: {best_c - best_100:.4f}")
