import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci_of_mean(arr, m: int = 1000, seed: int = 123, ci=(2.5, 97.5)):
    """
    Nonparametric bootstrap CI for the mean of arr.
    Returns (low, high).
    """
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        # With 1 sample, CI collapses to that value
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(m, n))
    means = arr[idx].mean(axis=1)

    low, high = np.percentile(means, list(ci))
    return float(low), float(high)


def clear_leaning_rate(
    input_csv: str,
    id_col: str = "ID",
    score_col: str = "score",
    decimals: int = 2,
    threshold: float = 0.10,
    bootstrap_m: int = 1000,
    seed: int = 123,
):
    """
    Reads a response CSV, groups scores by ID, rescales score to [-1, 1] via (d-4)/3,
    then uses bootstrap CI for the mean.
    A prompt is clear-leaning if CI is entirely outside [-threshold, threshold].
    Returns a dict with totals and percentage.
    """
    df = pd.read_csv(input_csv)

    # numeric scores
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col]).copy()

    # rescale 1..7 -> [-1, 1]
    df["score_rescaled"] = ((df[score_col] - 4.0) / 3.0).round(decimals)

    grouped = (
        df[[id_col, "score_rescaled"]]
        .groupby(id_col, sort=False)["score_rescaled"]
        .apply(list)
        .reset_index()
    )

    total = len(grouped)
    clear = 0
    unclear = 0

    for _, r in grouped.iterrows():
        arr = r["score_rescaled"]
        low, high = bootstrap_ci_of_mean(
            arr, m=bootstrap_m, seed=seed, ci=(2.5, 97.5)
        )
        # clear-leaning if CI fully above +threshold or fully below -threshold
        is_clear = (low > threshold) or (high < -threshold)
        if is_clear:
            clear += 1
        else:
            unclear += 1

    pct = 0.0 if total == 0 else (100.0 * clear / total)
    return {
        "input_csv": input_csv,
        "total_ids": total,
        "clear_ids": clear,
        "unclear_ids": unclear,
        "clear_rate_pct": pct,
    }


def plot_clear_leaning_rates(
    rules,
    rates_4b,
    rates_12b,
    out_png: str = "clear_leaning_rate.png",
    title: str = "Clear-leaning rate by wording rules (4B vs 12B)",
    figsize=(14, 6),
    title_size=18,
    label_size=18,
    tick_size=18,
    legend_size=15,
):
    """
    Make a horizontal scatter plot matching your provided figure style.
    """
    y = np.arange(len(rules))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(rates_4b, y, marker="o", label="gemma-3-4b-it")
    ax.scatter(rates_12b, y, marker="s", label="gemma-3-12b-it")

    ax.set_yticks(y)
    ax.set_yticklabels(rules, fontsize=tick_size)
    ax.invert_yaxis()  # make 'base' appear on top like your figure

    ax.set_xlim(0, 100)
    ax.set_xlabel("Clear-leaning responses (%)", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    ax.tick_params(axis="x", labelsize=tick_size)

    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower left", fontsize=legend_size)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[OK] Saved figure to: {out_png}")


def main():
    # ====== 1) 修改这里的路径为你本地实际 CSV 路径 ======
    # key 是图上的 wording rule 名称（y 轴顺序也由这里决定）
    # value 里分别填 4B/12B 的 response CSV
    RULE_TO_CSV = {
        "base": {
            "4b": "data/original_responses_4B.csv",
            "12b": "data/original_responses_12B.csv",
        },
        "negation": {
            "4b": "data/negation_responses_4B.csv",
            "12b": "data/negation_responses_12B.csv",
        },
        "opposite": {
            "4b": "data/opposite_responses_4B.csv",
            "12b": "data/opposite_responses_12B.csv",
        },
        "active-passive": {
            "4b": "data/active_passive_responses_4B.csv",
            "12b": "data/active_passive_responses_12B.csv",
        },
        "it-clefts": {
            "4b": "data/it-clefts_responses_4B.csv",
            "12b": "data/it-clefts_responses_12B.csv",
        },
        "wh-clefts": {
            "4b": "data/wh-clefts_responses_4B.csv",
            "12b": "data/wh-clefts_responses_12B.csv",
        },
        "SVC": {
            "4b": "data/SVC_responses_4B.csv",
            "12b": "data/SVC_responses_12B.csv",
        },
    }

    # ====== 2) 统计 clear-leaning rate ======
    summary_rows = []
    rules = list(RULE_TO_CSV.keys())
    rates_4b = []
    rates_12b = []

    for rule in rules:
        path_4b = RULE_TO_CSV[rule]["4b"]
        path_12b = RULE_TO_CSV[rule]["12b"]

        if not os.path.exists(path_4b):
            raise FileNotFoundError(f"Missing 4B file for rule '{rule}': {path_4b}")
        if not os.path.exists(path_12b):
            raise FileNotFoundError(f"Missing 12B file for rule '{rule}': {path_12b}")

        r4 = clear_leaning_rate(path_4b, seed=123)
        r12 = clear_leaning_rate(path_12b, seed=123)

        rates_4b.append(r4["clear_rate_pct"])
        rates_12b.append(r12["clear_rate_pct"])

        summary_rows.append(
            {
                "rule": rule,
                "model": "gemma-3-4b-it",
                "total_ids": r4["total_ids"],
                "clear_ids": r4["clear_ids"],
                "unclear_ids": r4["unclear_ids"],
                "clear_rate_pct": round(r4["clear_rate_pct"], 2),
                "csv": path_4b,
            }
        )
        summary_rows.append(
            {
                "rule": rule,
                "model": "gemma-3-12b-it",
                "total_ids": r12["total_ids"],
                "clear_ids": r12["clear_ids"],
                "unclear_ids": r12["unclear_ids"],
                "clear_rate_pct": round(r12["clear_rate_pct"], 2),
                "csv": path_12b,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    out_csv = "clear_leaning_rates_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote summary to: {out_csv}")
    print(summary_df)

    # ====== 3) 画图（样式对齐你给的图） ======
    plot_clear_leaning_rates(
        rules=rules,
        rates_4b=rates_4b,
        rates_12b=rates_12b,
        out_png="clear_leaning_rate.png",
        title="Clear-leaning rate by wording rules (4B vs 12B)",
    )


if __name__ == "__main__":
    main()
