import re
import os
import math
import pandas as pd
from typing import Dict, Tuple, List

files = {
    "original": "data/original_responses_4B.csv",
    "active_passive": "data/active_passive_responses_4B.csv",
    "it_cleft": "data/it-clefts_responses_4B.csv",
    "wh_cleft": "data/wh-clefts_responses_4B.csv",
    "SVC": "data/SVC_responses_4B.csv"
}

out_dir = "data/variance decomposition"

BASE_ID_RE = re.compile(r"^[a-z]{2}_[0-9]{1,2}")

def extract_base_id(id: str) -> str:
    id = id.strip()
    base_id = BASE_ID_RE.match(id)
    if base_id:
        return base_id.group()
    else:
        print(f"[warn] {id} caused an error when extracting base id")

def load_one(variant: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["ID", "score"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["base_id"] = df["ID"].map(extract_base_id)
    df["variant"] = variant
    return df[["ID", "base_id", "variant", "score"]]

def load_all(files: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for v,p in files.items():
        if not os.path.exists(p):
            print(f"[warn] {p} does not exist")
            continue
        df_v = load_one(v, p)
        frames.append(df_v)
        if v == "original":
            df_original = df_v.copy()
    
    df_all = pd.concat(frames, ignore_index=True)
    return df_all, df_original

def compute_as_per_rule(df_all: pd.DataFrame, base_variant: str = "original") -> Dict[str, float]:
    """
    For each rule variant v != base_variant:
      - keep only rows with variant in {base_variant, v}
      - compute per-(base_id, variant) mean score across samples
      - compute AS_rule = mean_s Var_over_variants(ybar_s,variant)  (with 2 variants -> (diff^2)/2)
    Returns: {variant_name: AS_rule_scalar}
    """
    variants = [v for v in df_all["variant"].dropna().unique().tolist() if v != base_variant]
    as_by_rule = {}

    for rule in variants:
        df_sub = df_all[df_all["variant"].isin([base_variant, rule])].copy()

        # mean across samples r for each (s, v)
        g_sv = (
            df_sub.groupby(["base_id", "variant"], sort=False)["score"]
            .mean()
            .rename("ybar_sv")
            .reset_index()
        )

        # pivot to get columns for original and rule
        wide = g_sv.pivot(index="base_id", columns="variant", values="ybar_sv")

        # keep only statements that have BOTH original and rule
        if base_variant not in wide.columns or rule not in wide.columns:
            as_by_rule[rule] = float("nan")
            continue

        pair = wide[[base_variant, rule]].dropna()

        if pair.shape[0] == 0:
            as_by_rule[rule] = float("nan")
            continue

        # With 2 variants, sample variance (ddof=1) equals (diff^2)/2
        diff = pair[base_variant] - pair[rule]
        as_rule = float(((diff * diff) / 2.0).mean())

        as_by_rule[rule] = as_rule

        print(f"AS({rule}): {as_rule:.6f}  (n_statements={pair.shape[0]})")

    return as_by_rule

if __name__ == "__main__":
    df_all, df_original = load_all(files)
    compute_as_per_rule(df_all, base_variant="original")
