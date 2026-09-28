import pickle
with open('output/phase6_audit_data.pkl', 'rb') as f:
    d = pickle.load(f)

raw_dict = d['raw_dict']
dev_gt = d['dev_gt']
retained = 0
total_gt = sum(len(gt) for gt in dev_gt.values())

for s1, cands in raw_dict.items():
    gt = set(dev_gt.get(s1, []))
    capped = set(c.matched_entity_id for c in sorted(cands, key=lambda x: len(x.channels), reverse=True)[:100])
    retained += len(gt & capped)

print(f"Buggy Recall (Cap 100): {retained/total_gt:.4f}")
