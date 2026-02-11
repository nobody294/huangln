import pandas as pd
import numpy as np
import json

def bootstrap_means(arr, m: int = 1000, seed: int = 123, decimals: int = 2):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(m, n))
    means_un = arr[idx].mean(axis=1)

    lower, upper = np.percentile(means_un, [2.5, 97.5])
    means = np.round(means_un, decimals)
    lower = round(float(lower), decimals)
    upper = round(float(upper), decimals)

    return means, (lower, upper)

def run(input_csv: str, output_csv: str, decimals: int = 2) -> None:
    df = pd.read_csv(input_csv)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()
    df["score_rescaled"] = ((df["score"] - 4.0) / 3.0).round(decimals)

    grouped = (
        df[["ID", "score_rescaled"]]
        .groupby("ID", sort=False)["score_rescaled"]
        .apply(list)
        .reset_index()
    )

    rows = []
    for _, r in grouped.iterrows():
        means, (low, high) = bootstrap_means(arr=r["score_rescaled"])
        if low > 0.10 or high < -0.10:
            continue
        else:
            rows.append({
                "ID": r["ID"],
                "score_list": json.dumps([float(x) for x in means]),
                "CI": json.dumps([float(low), float(high)])
            })
    
    out_df = pd.DataFrame(rows, columns=["ID", "score_list", "CI"])
    out_df.to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(out_df)} rows to: {output_csv}")

if __name__ == "__main__":
    run(input_csv="data/original_responses_4B.csv", output_csv="data/significance/original_4B_not_significant.csv")
    run(input_csv="data/original_responses_4B_qwen.csv", output_csv="data/significance/original_4B_not_significant_qwen.csv")
    run(input_csv="data/original_responses_12B.csv", output_csv="data/significance/original_12B_not_significant.csv")
    run(input_csv="data/original_responses_14B.csv", output_csv="data/significance/original_14B_not_significant.csv")

    run(input_csv="data/negation_responses_4B.csv", output_csv="data/significance/negation_4B_not_significant.csv")
    run(input_csv="data/negation_responses_4B_qwen.csv", output_csv="data/significance/negation_4B_not_significant_qwen.csv")
    run(input_csv="data/negation_responses_12B.csv", output_csv="data/significance/negation_12B_not_significant.csv")
    run(input_csv="data/negation_responses_14B.csv", output_csv="data/significance/negation_14B_not_significant.csv")

    run(input_csv="data/opposite_responses_4B.csv", output_csv="data/significance/opposite_4B_not_significant.csv")
    run(input_csv="data/opposite_responses_12B.csv", output_csv="data/significance/opposite_12B_not_significant.csv")
    run(input_csv="data/opposite_responses_4B_qwen.csv", output_csv="data/significance/opposite_4B_not_significant_qwen.csv")
    run(input_csv="data/opposite_responses_14B.csv", output_csv="data/significance/opposite_14B_not_significant.csv")

    run(input_csv="data/active_passive_responses_4B.csv", output_csv="data/significance/active_passive_4B_not_significant.csv")
    run(input_csv="data/active_passive_responses_12B.csv", output_csv="data/significance/active_passive_12B_not_significant.csv")

    run(input_csv="data/it-clefts_responses_4B.csv", output_csv="data/significance/it-clefts_4B_not_significant.csv")
    run(input_csv="data/it-clefts_responses_12B.csv", output_csv="data/significance/it-clefts_12B_not_significant.csv")

    run(input_csv="data/wh-clefts_responses_4B.csv", output_csv="data/significance/wh-clefts_4B_not_significant.csv")
    run(input_csv="data/wh-clefts_responses_12B.csv", output_csv="data/significance/wh-clefts_12B_not_significant.csv")

    run(input_csv="data/SVC_responses_4B.csv", output_csv="data/significance/SVC_4B_not_significant.csv")
    run(input_csv="data/SVC_responses_12B.csv", output_csv="data/significance/SVC_12B_not_significant.csv")
