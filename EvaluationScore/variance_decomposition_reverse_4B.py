import re
import os
import math
import pandas as pd
from typing import Dict, Tuple, List

files = {
    "original": "data/original_responses_4B.csv",
    "negation": "data/negation_responses_4B.csv",
    "opposite": "data/opposite_responses_4B.csv"
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

def mirror_likert_1_to_7(x: float) -> float:
    # 1<->7, 2<->6, 3<->5, 4 stays 4  ==>  x' = 8 - x
    return 8 - x

def load_one(variant: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["ID", "score"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if variant == "original":
        mask = df["score"].notna()
        df.loc[mask, "score"] = df.loc[mask, "score"].apply(mirror_likert_1_to_7)

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

def compute_variance_decomposition(df_all: pd.DataFrame, df_original: pd.DataFrame):
    g_v_id = (
        df_all.groupby(["variant", "ID"], sort=False)["score"]
        .agg(R="size",
             MU=lambda x: float(x.var(ddof=1)) if x.size > 1 else 0.0)
        .reset_index()
    )
    mu_by_variant = {
        v: sub[["ID", "MU"]]
        for v, sub in g_v_id.groupby("variant", sort=False)
    }

    g_sv = (
        df_all.groupby(["base_id", "variant"], sort=False)["score"]
        .mean()
        .rename("ybar_sv")
        .reset_index()
    )
    as_s = (
        g_sv.groupby("base_id", sort=False)["ybar_sv"]
        .agg(AS=lambda x: float(x.var(ddof=1)) if x.size > 1 else 0.0)
        .reset_index()
    )
    nvar = (
        g_sv.groupby("base_id", sort=False)["variant"]
        .nunique()
        .rename("n_variants")
        .reset_index()
    )
    as_df_raw = as_s.merge(nvar, on="base_id", how="outer")
    base_ids = df_original[["base_id"]].drop_duplicates()
    as_df = base_ids.merge(as_df_raw, on="base_id", how="left")
    as_df["AS"] = as_df["AS"].fillna(0.0)
    as_df["n_variants"] = as_df["n_variants"].fillna(0).astype(int)

    g_s = (
        g_sv.groupby("base_id", sort=False)["ybar_sv"]
        .mean()
        .rename("ybar_s")
        .reset_index()
    )
    ps_value = float(g_s["ybar_s"].var(ddof=1)) if g_s.shape[0] > 1 else 0.0

    return mu_by_variant, as_df, ps_value

def save_to_outputs(mu_by_variants: Dict, as_df: pd.DataFrame, ps_value: float, out_dir: str):
    for v, df_v in mu_by_variants.items():
        p = os.path.join(out_dir, f"{v}_MU_4B.csv")
        df_v.to_csv(p, index=False)
        print(f"Done for output {p}")
    
    p_as = os.path.join(out_dir, f"AS_4B_reverse.csv")
    as_df.to_csv(p_as, index=False)
    print(f"Done for output {p_as}")

    print(f"PS value: {ps_value:.6f}")

if __name__ == "__main__":
    df_all, df_original = load_all(files)
    mu_by_variant, as_df, ps_value = compute_variance_decomposition(df_all, df_original)
    save_to_outputs(mu_by_variant, as_df, ps_value, out_dir)
