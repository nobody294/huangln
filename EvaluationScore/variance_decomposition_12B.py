import re
import os
import math
import pandas as pd
from typing import Dict, Tuple, List

files = {
    "original": "data/original_responses_12B.csv",
    "active_passive": "data/active_passive_responses_12B.csv",
    "it_cleft": "data/it-clefts_responses_12B.csv",
    "wh_cleft": "data/wh-clefts_responses_12B.csv",
    "SVC": "data/SVC_responses_12B.csv"
}

BASE_ID_RE = re.compile(r"^[a-z]{2}_[0-9]{1,2}")

def extract_base_id(id_: str) -> str:
    """Extract base statement id (e.g., 'ab_12') from a longer ID string."""
    id_ = str(id_).strip()
    m = BASE_ID_RE.match(id_)
    if m:
        return m.group()
    else:
        print(f"[warn] {id_} caused an error when extracting base id")
        return None

def load_one(variant: str, path: str) -> pd.DataFrame:
    """Load one CSV into a normalized dataframe: ID, base_id, variant, score."""
    df = pd.read_csv(path)
    df = df[["ID", "score"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["base_id"] = df["ID"].map(extract_base_id)
    df["variant"] = variant

    df = df.dropna(subset=["score", "base_id", "ID"])
    return df[["ID", "base_id", "variant", "score"]]

def load_all(files: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for v, p in files.items():
        if not os.path.exists(p):
            print(f"[warn] {p} does not exist")
            continue
        frames.append(load_one(v, p))
    if not frames:
        raise RuntimeError("No input files were loaded. Please check paths.")
    return pd.concat(frames, ignore_index=True)

def compute_mu_per_variant(df_all: pd.DataFrame) -> pd.DataFrame:
    mu_prompt = (
        df_all.groupby(["variant", "ID"], sort=False)["score"]
        .agg(MU_prompt=lambda x: float(x.var(ddof=1)) if x.size > 1 else 0.0)
        .reset_index()
    )

    mu_variant = (
        mu_prompt.groupby("variant", sort=False)["MU_prompt"]
        .agg(MU="mean", n_prompts="size")
        .reset_index()
    )
    return mu_variant

def compute_pairwise_as_ps(df_all: pd.DataFrame, base_variant: str = "original") -> pd.DataFrame:
    variants = [v for v in df_all["variant"].dropna().unique().tolist() if v != base_variant]
    rows = []

    for rule in variants:
        df_sub = df_all[df_all["variant"].isin([base_variant, rule])].copy()

        g_sv = (
            df_sub.groupby(["base_id", "variant"], sort=False)["score"]
            .mean()
            .rename("ybar_sv")
            .reset_index()
        )

        wide = g_sv.pivot(index="base_id", columns="variant", values="ybar_sv")

        if base_variant not in wide.columns or rule not in wide.columns:
            rows.append({"rule": rule, "AS": float("nan"), "PS": float("nan"), "n_statements": 0})
            continue

        pair = wide[[base_variant, rule]].dropna()
        n_statements = int(pair.shape[0])
        if n_statements == 0:
            rows.append({"rule": rule, "AS": float("nan"), "PS": float("nan"), "n_statements": 0})
            continue

        diff = pair[base_variant] - pair[rule]
        as_rule = float(((diff * diff) / 2.0).mean())

        ybar_s = pair.mean(axis=1)
        ps_pair = float(ybar_s.var(ddof=1)) if n_statements > 1 else 0.0

        rows.append({"rule": rule, "AS": as_rule, "PS": ps_pair, "n_statements": n_statements})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_all = load_all(files)

    mu_variant = compute_mu_per_variant(df_all)
    print("=== MU per variant (one scalar per wording rule / variant) ===")
    for _, r in mu_variant.iterrows():
        print(f"MU({r['variant']}): {float(r['MU']):.6f}  (n_prompts={int(r['n_prompts'])})")

    pair_metrics = compute_pairwise_as_ps(df_all, base_variant="original")
    print("=== Pairwise metrics vs original (per rule) ===")
    for _, r in pair_metrics.iterrows():
        print(
            f"rule={r['rule']:<15}  "
            f"AS={float(r['AS']):.6f}  "
            f"PS(original+rule)={float(r['PS']):.6f}  "
            f"(n_statements={int(r['n_statements'])})"
        )
