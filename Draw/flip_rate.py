import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.10

POLARITY_REVERSING = {"negation", "opposite"}

RULE_ORDER = ["negation", "opposite", "active-passive", "it-clefts", "wh-clefts", "SVC"]

PATHS = {
    "gemma-3-4b-it": {
        "original": "data/original_CI_4B.csv",
        "variants": {
            "negation": "data/negation_CI_4B.csv",
            "opposite": "data/opposite_CI_4B.csv",
            "active-passive": "data/active_passive_CI_4B.csv",
            "it-clefts": "data/it-clefts_CI_4B.csv",
            "wh-clefts": "data/wh-clefts_CI_4B.csv",
            "SVC": "data/SVC_CI_4B.csv",
        },
    },
    "gemma-3-12b-it": {
        "original": "data/original_CI_12B.csv",
        "variants": {
            "negation": "data/negation_CI_12B.csv",
            "opposite": "data/opposite_CI_12B.csv",
            "active-passive": "data/active_passive_CI_12B.csv",
            "it-clefts": "data/it-clefts_CI_12B.csv",
            "wh-clefts": "data/wh-clefts_CI_12B.csv",
            "SVC": "data/SVC_CI_12B.csv",
        },
    },
}

# 输出
OUT_PNG = "flip_rate.png"
OUT_SUMMARY_CSV = "flip_rate_summary.csv"


# ============= 工具函数 =============
def parse_ci(ci_str: str):
    low, high = json.loads(ci_str)
    return float(low), float(high)


def side_from_ci(ci, thr=THRESHOLD):
    if ci is None:
        return None
    low, high = ci
    if low > thr:
        return "pos"
    if high < -thr:
        return "neg"
    return None


def extract_base_id(id_str: str):
    s = str(id_str)
    m = re.match(r'^[a-z]{2}_[0-9]{1,2}', s)
    return m.group()


def load_ci_file(path: str):
    df = pd.read_csv(path).copy()

    # 基本列检查
    need_cols = {"ID", "CI"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{path}] Missing columns: {missing}. Found: {list(df.columns)}")

    df["CI_tuple"] = df["CI"].apply(parse_ci)
    df["side"] = df["CI_tuple"].apply(lambda t: side_from_ci(t, THRESHOLD))
    df["base_id"] = df["ID"].apply(extract_base_id)

    return df[["ID", "base_id", "CI", "CI_tuple", "side"]]


def is_flip(side_orig: str, side_var: str, rule_name: str):
    """
    flip = violates expected polarity relation. :contentReference[oaicite:3]{index=3}
    - polarity preserving: expectation invariance -> flip if different polarity
    - polarity reversing: expectation inversion -> flip if same polarity
    只在两边都 clear (pos/neg) 时调用。
    """
    if rule_name in POLARITY_REVERSING:
        # expected inversion -> violation means same polarity
        return side_orig == side_var
    else:
        # expected invariance -> violation means opposite polarity
        return {side_orig, side_var} == {"pos", "neg"}


def flip_rate_for_rule(original_ci_path: str, variant_ci_path: str, rule_name: str):
    """
    flip rate 只在 clear-leaning base-variant pairs 上计算：两边 side 都是 pos/neg 才 eligible。:contentReference[oaicite:4]{index=4}
    """
    df_o = load_ci_file(original_ci_path).set_index("base_id")
    df_v = load_ci_file(variant_ci_path).set_index("base_id")

    common = df_o.index.intersection(df_v.index)

    eligible = 0
    flips = 0
    for bid in common:
        o = df_o.loc[bid]
        v = df_v.loc[bid]
        if (o["side"] in {"pos", "neg"}) and (v["side"] in {"pos", "neg"}):
            eligible += 1
            if is_flip(o["side"], v["side"], rule_name):
                flips += 1

    rate = 0.0 if eligible == 0 else 100.0 * flips / eligible
    return eligible, flips, rate


def plot_flip_rates(
    rules,
    rate_4b,
    rate_12b,
    out_png=OUT_PNG,
    title: str = "Flip rate by wording rules (4B vs 12B)",
    figsize=(14, 6),
    title_size=18,
    label_size=18,
    tick_size=18,
    legend_size=15,
):
    # 画法对齐你截图：横向散点 + grid + invert_y
    y = np.arange(len(rules))

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(rate_4b, y, marker="o", label="gemma-3-4b-it")
    ax.scatter(rate_12b, y, marker="s", label="gemma-3-12b-it")

    ax.set_yticks(y)
    ax.set_yticklabels(rules, fontsize=tick_size)
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.set_xlabel("Flip Rate (%)", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    ax.tick_params(axis="x", labelsize=tick_size)

    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", fontsize=legend_size)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[OK] Saved: {out_png}")


def main():
    # 路径存在性检查
    for model, conf in PATHS.items():
        if not os.path.exists(conf["original"]):
            raise FileNotFoundError(f"[{model}] missing original CI: {conf['original']}")
        for rule in RULE_ORDER:
            p = conf["variants"].get(rule)
            if p is None:
                raise ValueError(f"[{model}] missing path for rule: {rule}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"[{model}] missing variant CI for {rule}: {p}")

    summary = []
    rates = {m: [] for m in PATHS.keys()}

    for model, conf in PATHS.items():
        orig = conf["original"]
        for rule in RULE_ORDER:
            varp = conf["variants"][rule]
            eligible, flips, rate = flip_rate_for_rule(orig, varp, rule)
            rates[model].append(rate)
            summary.append({
                "model": model,
                "rule": rule,
                "eligible_pairs": eligible,
                "flip_pairs": flips,
                "flip_rate_pct": round(rate, 2),
                "original_ci": orig,
                "variant_ci": varp,
                "expected_relation": "inversion" if rule in POLARITY_REVERSING else "invariance",
            })
            print(f"[{model}] {rule}: eligible={eligible}, flips={flips}, rate={rate:.2f}%")

    # summary csv
    out_df = pd.DataFrame(summary)
    out_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[OK] Wrote summary: {OUT_SUMMARY_CSV}")

    # plot
    plot_flip_rates(
        rules=RULE_ORDER,
        rate_4b=rates["gemma-3-4b-it"],
        rate_12b=rates["gemma-3-12b-it"],
        out_png=OUT_PNG,
    )


if __name__ == "__main__":
    main()
