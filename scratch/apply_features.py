import re
import os

addr_feat_path = "src/business_entity_resolution/features/address_features.py"
with open(addr_feat_path, "r") as f:
    addr_feat = f.read()

new_addr_feat = addr_feat.replace(
    'features["address_token_count_diff"] = float(abs(len(toks1) - len(toks2)))',
    '''features["address_token_count_diff"] = float(abs(len(toks1) - len(toks2)))
    
    import re
    def extract_street_num(raw: str):
        if not raw: return ""
        # match first token starting with a digit
        for t in raw.replace(",", " ").split():
            if t and t[0].isdigit():
                return t.strip(".,;:").upper()
        return ""
        
    num_ext1 = extract_street_num(addr1.raw)
    num_ext2 = extract_street_num(addr2.raw)
    
    def extract_numeric_comp(num_str: str):
        if not num_str: return ""
        return "".join(re.findall(r"\\d+", num_str))
        
    num_comp1 = extract_numeric_comp(num_ext1)
    num_comp2 = extract_numeric_comp(num_ext2)
    
    def get_num_tokens(raw: str):
        if not raw: return set()
        return set(re.findall(r"\\d+", raw))
        
    ntoks1 = get_num_tokens(addr1.raw)
    ntoks2 = get_num_tokens(addr2.raw)
'''
)

new_addr_feat = new_addr_feat.replace(
    '''    # If either address is missing, comparisons are 0.0 (safe defaults)
    if is_empty1 or is_empty2:
        features["address_normalized_exact_match"] = 0.0
        features["address_compact_exact_match"] = 0.0
        features["address_street_number_match"] = 0.0
        features["address_street_number_both_present"] = 0.0
        features["address_street_number_mismatch"] = 0.0
        features["address_postal_exact_match"] = 0.0
        features["address_postal_prefix_match"] = 0.0
        features["address_postal_both_present"] = 0.0
        features["address_postal_mismatch"] = 0.0
        features["address_street_stem_match"] = 0.0
        features["address_token_overlap_count"] = 0.0
        features["address_token_jaccard"] = 0.0
        features["address_levenshtein_sim"] = 0.0
        return features''',
    '''    # If either address is missing, comparisons are 0.0 (safe defaults)
    if is_empty1 or is_empty2:
        features["address_normalized_exact_match"] = 0.0
        features["address_compact_exact_match"] = 0.0
        features["address_street_number_match"] = 0.0
        features["address_street_number_both_present"] = 0.0
        features["address_street_number_mismatch"] = 0.0
        features["address_postal_exact_match"] = 0.0
        features["address_postal_prefix_match"] = 0.0
        features["address_postal_both_present"] = 0.0
        features["address_postal_mismatch"] = 0.0
        features["address_street_stem_match"] = 0.0
        features["address_token_overlap_count"] = 0.0
        features["address_token_jaccard"] = 0.0
        features["address_levenshtein_sim"] = 0.0
        
        # New features defaults for missing
        features["address_exact_street_number_match"] = -1.0
        features["address_street_number_numeric_match"] = -1.0
        features["address_postal_code_exact_match"] = -1.0
        features["address_postal_code_prefix_match"] = -1.0
        features["address_numeric_token_jaccard"] = -1.0
        features["address_number_consistency"] = -1.0
        return features'''
)

new_addr_feat = new_addr_feat.replace(
    '''    features["address_levenshtein_sim"] = round(levenshtein_similarity(norm1, norm2), cfg.float_precision)
    
    return features''',
    '''    features["address_levenshtein_sim"] = round(levenshtein_similarity(norm1, norm2), cfg.float_precision)

    # A. EXACT_STREET_NUMBER_MATCH
    if num_ext1 and num_ext2:
        features["address_exact_street_number_match"] = 1.0 if num_ext1 == num_ext2 else 0.0
    else:
        features["address_exact_street_number_match"] = -1.0
        
    # B. STREET_NUMBER_NUMERIC_MATCH
    if num_comp1 and num_comp2:
        features["address_street_number_numeric_match"] = 1.0 if num_comp1 == num_comp2 else 0.0
    else:
        features["address_street_number_numeric_match"] = -1.0
        
    # C. POSTAL_CODE_EXACT_MATCH
    if p1 and p2:
        features["address_postal_code_exact_match"] = 1.0 if p1 == p2 else 0.0
    else:
        features["address_postal_code_exact_match"] = -1.0
        
    # D. POSTAL_CODE_PREFIX_MATCH
    if both_have_postal and prefix1 and prefix2:
        features["address_postal_code_prefix_match"] = 1.0 if prefix1 == prefix2 else 0.0
    else:
        features["address_postal_code_prefix_match"] = -1.0
        
    # E. NUMERIC_TOKEN_JACCARD
    if ntoks1 and ntoks2:
        intersection = ntoks1.intersection(ntoks2)
        union = ntoks1.union(ntoks2)
        features["address_numeric_token_jaccard"] = round(len(intersection) / len(union), cfg.float_precision)
    else:
        features["address_numeric_token_jaccard"] = -1.0
        
    # F. ADDRESS_NUMBER_CONSISTENCY
    # 3 = both street number numeric and postal match
    # 2 = one matches, other unavailable
    # 1 = one matches, other mismatches
    # 0 = both mismatch
    # -1 = unavailable entirely
    
    s_match = features["address_street_number_numeric_match"]
    p_match = features["address_postal_code_exact_match"]
    
    if s_match == -1.0 and p_match == -1.0:
        c_score = -1.0
    elif s_match == 1.0 and p_match == 1.0:
        c_score = 3.0
    elif (s_match == 1.0 and p_match == -1.0) or (p_match == 1.0 and s_match == -1.0):
        c_score = 2.0
    elif (s_match == 1.0 and p_match == 0.0) or (p_match == 1.0 and s_match == 0.0):
        c_score = 1.0
    elif s_match == 0.0 and p_match == 0.0:
        c_score = 0.0
    else:
        # e.g., one is mismatch (0.0), one is unavailable (-1.0)
        c_score = 0.0
        
    features["address_number_consistency"] = c_score

    return features'''
)

with open(addr_feat_path, "w") as f:
    f.write(new_addr_feat)

feat_cfg_path = "src/business_entity_resolution/features/feature_config.py"
with open(feat_cfg_path, "r") as f:
    feat_cfg = f.read()

new_feat_cfg = feat_cfg.replace(
    '''                "address_token_count_diff",
                "address_length_diff",
            ])''',
    '''                "address_token_count_diff",
                "address_length_diff",
                "address_exact_street_number_match",
                "address_street_number_numeric_match",
                "address_postal_code_exact_match",
                "address_postal_code_prefix_match",
                "address_numeric_token_jaccard",
                "address_number_consistency",
            ])'''
)

with open(feat_cfg_path, "w") as f:
    f.write(new_feat_cfg)
