import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_csv_path      = "data/original_statements.csv"
active_passive_csv_path  = "data/active_passive_variants.csv"
SVC_csv_path  = "data/SVC_variants.csv"
output_csv_path    = "data/combine_active_passive_SVC_variants.csv"

model_name = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter for logical variants of short English statements. "
    "You will be given a BASE statement, its ACTIVE/PASSIVE CONVERSION variant, and its SUPPORT VERB CONSTRUCTION variant. "
    "Your only job is to construct a COMBINE variant that integrates BOTH types of changes. "
    "Intended meanings: "
    "- The ACTIVE/PASSIVE CONVERSION variant is formed by implementing only a voice transformation (Active<->Passive). "
    "- The SUPPORT VERB CONSTRUCTION variant is formed by transform the base statement into this structure: [SUPPORT_VERB] + [DEVERBAL_NOUN] (+ minimal required preposition) (+ original complements). "
    "- The COMBINE variant should: "
    "keep the voice transformation of the ACTIVE/PASSIVE CONVERSION variant, and "
    "keep the SUPPORT VERB CONSTRUCTION changes of the SUPPORT VERB CONSTRUCTION variant."
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "active/passive variant": "Stronger financial support should be provided to unemployed workers by the state.",
        "support verb construction variant": "The state should make provision for stronger financial support to unemployed workers.",
        "combine variant": "",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "active/passive variant": "Member States that violate the EU deficit rules should be rigorously punished by the EU.",
        "support verb construction variant": "The EU should rigorously impose punishment on Member States that violate the EU deficit rules.",
        "combine variant": "Member States that violate the EU deficit rules should be rigorously imposed punishment by the EU.",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "active/passive variant": "The government should tax bank and stock market gains more heavily.",
        "support verb construction variant": "Bank and stock market gains should be given heavier taxation.",
        "combine variant": "The government should impose heavier taxation on bank and stock market gains.",
    },
    {
        "base": "In European Parliament elections, EU citizens should be allowed to cast a vote for a party or candidate from any other Member State.",
        "active/passive variant": "In European Parliament elections, a vote should be allowed to be cast by EU citizens for a party or candidate from any other Member State.",
        "support verb construction variant": "In European Parliament elections, EU citizens should be given permission to cast a vote for a party or candidate from any other Member State.",
        "combine variant": "In European Parliament elections, permission to cast a vote should be given to EU citizens for a party or candidate from any other Member State.",
    },
    {
        "base": "The legalisation of same sex marriages is a good thing.",
        "active/passive variant": "null",
        "support verb construction variant": "null",
        "combine variant": "null",
    },
    {
        "base": "The legalisation of the personal use of soft drugs is to be welcomed.",
        "active/passive variant": "The government is to welcome the legalisation of the personal use of soft drugs.",
        "support verb construction variant": "null",
        "combine variant": "null",
    },
]

def render_fewshots_block(shots):
    lines = ["Few-shot exemplars (follow style strictly):"]
    for s in shots:
        lines.append(
            f"- Base: {s['base']}\n"
            f"- Active/Passive Conversion variant: {s['active/passive variant']}\n"
            f"- Support Verb Construction variant: {s['support verb construction variant']}\n"
            f"- Combine variant: {s['combine variant']}"
        )
    return "\n".join(lines)

def build_user_prompt(base: str, actpas: str, svc: str, fewshots_text: str) -> str:
    schema = """{
    "base": "<copy the base text exactly>",
    "active/passive_variant": "<copy the active/passive variant exactly>",
    "support_verb_construction_variant": "<copy the support verb construction variant exactly>",
    "combine_variant": {
        "text": "...",
        "not_applicable": false,
    }
    }"""
    return f"""Task: Construct a COMBINE variant from a base, an active/passive variant, and a support verb construction variant.

    Hard constraints (follow strictly):
    1) Do NOT add new content, explanations, or paraphrases beyond support verb construction and active/passive conversion.
    2) The COMBINE variant must be a single grammatical English sentence.
    3) If a well-formed combine variant cannot be constructed, set not_applicable = true.

    Output format (SINGLE JSON only, no extra text):
    {schema}

    {fewshots_text}

    Base statement: {base}

    Active/Passive Conversion variant: {actpas}

    Support Verb Construction variant: {svc}
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
    df_actpas  = pd.read_csv(active_passive_csv_path)
    df_svc  = pd.read_csv(SVC_csv_path)

    df_base = add_prefix_column(df_base, "base")
    df_actpas  = add_prefix_column(df_actpas,  "active/passive")
    df_svc  = add_prefix_column(df_svc,  "support verb construction")

    base_small = df_base[["prefix", "statement"]].rename(columns={"statement": "base_stmt"})
    actpas_small  = df_actpas[["prefix", "statement"]].rename(columns={"statement": "actpas_stmt"})
    svc_small  = df_svc[["prefix", "statement"]].rename(columns={"statement": "svc_stmt"})

    merged = (
        base_small
        .merge(actpas_small, on="prefix", how="inner")
        .merge(svc_small, on="prefix", how="inner")
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
        actpas_str  = str(row["actpas_stmt"])
        svc_str  = str(row["svc_stmt"])

        user_prompt = build_user_prompt(base_str, actpas_str, svc_str, fewshots_text)
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
