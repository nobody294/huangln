import re
import os
import json
import pandas as pd

original_csv = "data/original_CI_4B.csv"
variant_csvs = {
    "negation": "data/negation_CI_4B.csv",
    "opposite": "data/opposite_CI_4B.csv",
    "active_passive": "data/active_passive_CI_4B.csv",
    "it-clefts": "data/it-clefts_CI_4B.csv",
    "wh-clefts": "data/wh-clefts_CI_4B.csv",
    "SVC": "data/SVC_CI_4B.csv"
}

out_dir = "data/flip rate"

threshold = 0.10

same_side_flip = {"negation", "opposite"}

def parse_ci(ci_str):
    low, high = json.loads(ci_str)
    return (float(low), float(high))

def side_from_ci(ci, thr=threshold):
    if ci is None:
        return None
    low, high = ci
    if low > +thr:
        return "pos"
    if high < -thr:
        return "neg"
    return None

def extract_base_id(id_str):
    s = str(id_str)
    m = re.match(r'^[a-z]{2}_[0-9]{1,2}', s)
    return m.group()

def load_ci_file(path):
    df = pd.read_csv(path)
    df = df.copy()
    df["CI_tuple"] = df["CI"].apply(parse_ci)
    df["side"] = df["CI_tuple"].apply(lambda t: side_from_ci(t, threshold))
    df["base_id"] = df["ID"].apply(extract_base_id)
    return df[["ID", "base_id", "score_list", "CI", "CI_tuple", "side"]]

def decide_flip(side_orig, side_var, variant_name):
    if (side_orig is None) or (side_var is None):
        return False
    if variant_name in same_side_flip:
        return (side_orig == side_var)
    else:
        return ({side_orig, side_var} == {"pos", "neg"})

df_original = load_ci_file(original_csv)
original_by_base = df_original.set_index("base_id")
print("Loaded original csv.")

for vname, vpath in variant_csvs.items():
    df_variant = load_ci_file(vpath)
    variant_by_base = df_variant.set_index("base_id")

    common_base_ids = original_by_base.index.intersection(variant_by_base.index)
    flip_rows = []
    eligible_count = 0
    for base_id in common_base_ids:
        o = original_by_base.loc[base_id]
        v = variant_by_base.loc[base_id]

        if (o["side"] in {"pos", "neg"}) and (v["side"] in {"pos", "neg"}):
            eligible_count += 1
            if decide_flip(o["side"], v["side"], vname):
                flip_rows.append({
                    "ID": v["ID"],
                    "score_list": v["score_list"],
                    "CI": v["CI"]
                })
    
    df_flip = pd.DataFrame(flip_rows, columns=["ID", "score_list", "CI"])
    out_name = f"{vname}_flip_4B.csv"
    out_path = os.path.join(out_dir, out_name)
    df_flip.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[ok] {vname}: pairs={eligible_count}, flips={len(df_flip)} -> {out_path}")
