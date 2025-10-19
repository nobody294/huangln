import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_id"] = df["ID"].str.rsplit("_", n=1, expand=True)[0]
    return df

def compute_w1(original_csv: str, variant_csv: str, output_csv: str) -> None:
    df_original = pd.read_csv(original_csv, dtype={"ID": str})
    df_variant = pd.read_csv(variant_csv, dtype={"ID": str})

    df_original = normalize_id(df_original)
    df_variant = normalize_id(df_variant)
    
    df_original["score"] = pd.to_numeric(df_original["score"], errors="coerce")
    df_variant["score"] = pd.to_numeric(df_variant["score"], errors="coerce")
    df_original = df_original.dropna(subset=["score"])
    df_variant = df_variant.dropna(subset=["score"])

    list_original = df_original.groupby("base_id", sort=False)["score"].apply(list)
    list_variant = df_variant.groupby("base_id", sort=False)["score"].apply(list)

    common_ids = list_original.index.intersection(list_variant.index)

    results = []
    for id in common_ids:
        x = list_original.loc[id]
        y = list_variant.loc[id]
        if len(x) != 30 or len(y) != 30:
            print(f"[warn] {id} does not have 30 responses!")
            continue
        w1 = wasserstein_distance(x, y)
        results.append({"ID": id, "W1": w1})
    
    pd.DataFrame(results, columns=["ID", "W1"]).to_csv(output_csv, index=False, encoding="utf-8")
    print("W1 score calculation complete.")

if __name__ == "__main__":
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/negation_responses_4B.csv",
               output_csv="data/negation_W1_4B.csv")
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/opposite_responses_4B.csv",
               output_csv="data/opposite_W1_4B.csv")
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/active_passive_responses_4B.csv",
               output_csv="data/active_passive_W1_4B.csv")
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/it-clefts_responses_4B.csv",
               output_csv="data/it-clefts_W1_4B.csv")
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/wh-clefts_responses_4B.csv",
               output_csv="data/wh-clefts_W1_4B.csv")
    compute_w1(original_csv="data/original_responses_4B.csv", 
               variant_csv="data/SVC_responses_4B.csv",
               output_csv="data/SVC_W1_4B.csv")
    
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/negation_responses_12B.csv",
               output_csv="data/negation_W1_12B.csv")
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/opposite_responses_12B.csv",
               output_csv="data/opposite_W1_12B.csv")
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/active_passive_responses_12B.csv",
               output_csv="data/active_passive_W1_12B.csv")
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/it-clefts_responses_12B.csv",
               output_csv="data/it-clefts_W1_12B.csv")
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/wh-clefts_responses_12B.csv",
               output_csv="data/wh-clefts_W1_12B.csv")
    compute_w1(original_csv="data/original_responses_12B.csv", 
               variant_csv="data/SVC_responses_12B.csv",
               output_csv="data/SVC_W1_12B.csv")
