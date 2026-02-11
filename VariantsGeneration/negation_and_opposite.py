import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_csv_path      = "data/original_statements.csv"
negation_csv_path  = "data/negation_variants.csv"
opposite_csv_path  = "data/opposite_variants.csv"
output_csv_path    = "data/combine_negation_opposite_variants.csv"

model_name = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter for logical variants of short English statements. "
    "You will be given a BASE statement, its NEGATION variant, and its OPPOSITE variant. "
    "Your only job is to construct a COMBINE variant that integrates BOTH types of changes. "
    "Intended meanings: "
    "- The NEGATION variant is formed by logically negating the base (e.g. inserting or removing 'not'). "
    "- The OPPOSITE variant is formed by changing some lexical items to semantic opposites. "
    "- The COMBINE variant should: "
    "keep the NEGATED changes of the NEGATION variant, and "
    "keep the LEXICAL opposite changes of the OPPOSITE variant."
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "negation variant": "The state should not provide stronger financial support to unemployed workers.",
        "opposite variant": "The state should provide weaker financial support to unemployed workers.",
        "combine variant": "The state should not provide weaker financial support to unemployed workers.",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "negation variant": "The EU should not rigorously punish Member States that violate the EU deficit rules.",
        "opposite variant": "The EU should rigorously punish Member States that respect the EU deficit rules.",
        "combine variant": "The EU should not rigorously punish Member States that respect the EU deficit rules.",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "negation variant": "Bank and stock market gains should not be taxed more heavily.",
        "opposite variant": "Bank and stock market gains should be taxed less heavily.",
        "combine variant": "Bank and stock market gains should not be taxed less heavily.",
    },
    {
        "base": "In European Parliament elections, EU citizens should be allowed to cast a vote for a party or candidate from any other Member State.",
        "negation variant": "In European Parliament elections, EU citizens should not be allowed to cast a vote for a party or candidate from any other Member State.",
        "opposite variant": "In European Parliament elections, EU citizens should be forbidden to cast a vote for a party or candidate from any other Member State.",
        "combine variant": "In European Parliament elections, EU citizens should not be forbidden to cast a vote for a party or candidate from any other Member State.",
    },
    {
        "base": "The legalisation of same sex marriages is a good thing.",
        "negation variant": "The legalisation of same sex marriages is not a good thing.",
        "opposite variant": "The legalisation of same sex marriages is a bad thing.",
        "combine variant": "The legalisation of same sex marriages is not a bad thing.",
    },
    {
        "base": "The legalisation of the personal use of soft drugs is to be welcomed.",
        "negation variant": "The legalisation of the personal use of soft drugs is not to be welcomed.",
        "opposite variant": "The legalisation of the personal use of soft drugs is to be condemned.",
        "combine variant": "The legalisation of the personal use of soft drugs is not to be condemned.",
    },
]

def render_fewshots_block(shots):
    lines = ["Few-shot exemplars (follow style strictly):"]
    for s in shots:
        lines.append(
            f"- Base: {s['base']}\n"
            f"- Negation variant: {s['negation variant']}\n"
            f"- Opposite variant: {s['opposite variant']}\n"
            f"- Combine variant: {s['combine variant']}"
        )
    return "\n".join(lines)

def build_user_prompt(base: str, neg: str, opp: str, fewshots_text: str) -> str:
    schema = """{
    "base": "<copy the base text exactly>",
    "negation_variant": "<copy the negation variant exactly>",
    "opposite_variant": "<copy the opposite variant exactly>",
    "combine_variant": {
        "text": "...",
        "not_applicable": false,
    }
    }"""
    return f"""Task: Construct a COMBINE variant from a base, a negation variant, and an opposite variant.

    Hard constraints:
    1) Do NOT add new content, explanations, or paraphrases.
    2) The COMBINE variant must be a single grammatical English sentence.
    3) If a well-formed combine variant cannot be constructed, set not_applicable = true.

    Output format (SINGLE JSON only, no extra text):
    {schema}

    {fewshots_text}

    Base statement: {base}

    Negation variant: {neg}

    Opposite variant: {opp}
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(s: str):
    m = JSON_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def pick_variant_text(obj):
    if not isinstance(obj, dict):
        return None
    combine = obj.get("combine_variant")
    
    if isinstance(combine, dict):
        if combine.get("not_applicable", False):
            return None
        if "text" in combine:
            return str(combine["text"]).strip()
        return None
    
    if isinstance(combine, str):
        return combine.strip()

def replace_suffix(id_str: str, new_suffix: str) -> str:
    m = re.match(r"^([^_]+_[^_]+)_[0-9]{7}$", id_str)
    if m:
        return f"{m.group(1)}_{new_suffix}"
    
    if "_" in id_str:
        head = "_".join(id_str.split("_")[:2])
        return f"{head}_{new_suffix}"
    return f"{id_str}_{new_suffix}"

def chat_complete(model, tokenizer, system_prompt, user_prompt,
                  max_new_tokens=4096):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5
        )
    output_ids = out[0][len(inputs.input_ids[0]):].tolist()
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    return content.strip()

def add_prefix_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if list(df.columns[:2]) != ["ID", "statement"]:
        raise ValueError(f"{source_name}: first two columns must be ID and statement")

    df = df[["ID", "statement"]].copy()
    df["prefix"] = df["ID"].astype(str).str.rsplit("_", n=1).str[0]

    dup_mask = df["prefix"].duplicated(keep=False)
    if dup_mask.any():
        dup_prefixes = sorted(df.loc[dup_mask, "prefix"].unique())
        print(f"[warn] {source_name}: found duplicated prefixes, will keep the first occurrence:")
        for p in dup_prefixes:
            print(f"  - {p}", file=sys.stderr)
        df = df.drop_duplicates(subset="prefix", keep="first")

    return df

def make_combine_id(prefix: str) -> str:
    return f"{prefix}_0110000"

def run():
    fewshots_text = render_fewshots_block(BUILTIN_FEWSHOTS)

    df_base = pd.read_csv(base_csv_path)
    df_neg  = pd.read_csv(negation_csv_path)
    df_opp  = pd.read_csv(opposite_csv_path)

    df_base = add_prefix_column(df_base, "base")
    df_neg  = add_prefix_column(df_neg,  "negation")
    df_opp  = add_prefix_column(df_opp,  "opposite")

    base_small = df_base[["prefix", "statement"]].rename(columns={"statement": "base_stmt"})
    neg_small  = df_neg[["prefix", "statement"]].rename(columns={"statement": "neg_stmt"})
    opp_small  = df_opp[["prefix", "statement"]].rename(columns={"statement": "opp_stmt"})

    merged = (
        base_small
        .merge(neg_small, on="prefix", how="inner")
        .merge(opp_small, on="prefix", how="inner")
    )

    if merged.empty:
        print("[error] No common ID prefixes found across the three CSV files.", file=sys.stderr)
        return

    total_groups = len(merged)
    print(f"[info] Found {total_groups} sentence groups with matching ID prefixes.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    out_rows = []
    success_count = 0

    for i, row in merged.iterrows():
        prefix   = str(row["prefix"])
        base_str = str(row["base_stmt"])
        neg_str  = str(row["neg_stmt"])
        opp_str  = str(row["opp_stmt"])

        user_prompt = build_user_prompt(base_str, neg_str, opp_str, fewshots_text)
        raw = chat_complete(model, tokenizer, SYSTEM_PROMPT, user_prompt)
        obj = extract_first_json(raw)
        variant = pick_variant_text(obj)

        if not variant:
            print(raw)
            print(f"[warn] Group {i} (prefix={prefix}) JSON/combine_variant failed, setting statement=None.",
                  file=sys.stderr)
            variant = None
        else:
            print(raw)
            print(f"Group {i} (prefix={prefix}) JSON/combine_variant succeed.")
            success_count += 1

        new_id = make_combine_id(prefix)
        out_rows.append({"ID": new_id, "statement": variant})

    pd.DataFrame(out_rows, columns=["ID", "statement"]).to_csv(
        output_csv_path, index=False, encoding="utf-8"
    )
    print(f"{success_count} / {total_groups} groups have valid combine variants.")
    print(f"[done] Wrote: {output_csv_path}")

if __name__ == "__main__":
    run()
