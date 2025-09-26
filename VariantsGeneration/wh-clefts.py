import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CSV file paths
input_csv_dir  = "data/original_statements.csv"
output_csv_dir = "data/wh-clefts_variants.csv"

model_name = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter. "
    "Your only job is to transform the base statement into a Wh-cleft (pseudo-cleft) construction: "
    "What/Who/Where/When + [CLause with a gap] + is/are/was/were + [FOCUS]. "
    "Truth conditions must be preserved. Do not introduce or remove content. Do not paraphrase. "
    "Keep the original determiner type and genericity. "
    "Preserve scope of negation, modals, quantifiers, tense/aspect, numbers, named entities, and PPs (time/place). "
    "The [FOCUS] must be a contiguous span copied verbatim from the base (NP/PP preferred; VP allowed if necessary). "
    "Generate in English only. "
    "Output a single JSON object exactly matching the schema."
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "variant": "What the state should provide to unemployed workers is stronger financial support.",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "variant": "What the EU should rigorously punish is Member States that violate the EU deficit rules.",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "variant": "What should be taxed more heavily are bank and stock market gains.",
    },
    {
        "base": "In European Parliament elections, EU citizens should be allowed to cast a vote for a party or candidate from any other Member State.",
        "variant": "What EU citizens should be allowed to cast in European Parliament elections is a vote for a party or candidate from any other Member State.",
    },
]

def render_fewshots_block(shots):
    lines = ["Few-shot exemplars (follow style strictly):"]
    for s in shots:
        lines.append(f"- Base: {s['base']}\n  - wh_cleft variant: {s['variant']}")
    return "\n".join(lines)

def build_user_prompt(base: str, fewshots_text: str) -> str:
    schema = """{
  "base": "<copy the base exactly>",
  "variants": {
      "text": "...",
      "not_applicable": false,
      "reason": none
  }
}"""
    return f"""Task: Convert the base statement into a Wh-cleft variant.

Hard constraints (follow strictly):
1) Use canonical Wh-cleft form: What/Who/Where/When + [CLause with a gap] + is/are/was/were + [FOCUS]. Match the tense to the base.
2) WH choice: 'what' for things or VP-gaps (default), 'who' ONLY for people, 'where' for places, 'when' for times. Do not use 'why'.
3) Do not new content if it is unnecessary.
4) [FOCUS] must be a contiguous verbatim span from the base. Prefer NP or PP. VP allowed only if it is a contiguous phrase copied verbatim.
5) [FOCUS] preference order: object NP/PP > adjunct PP (time/place) > subject NP > VP.
6) Keep the original article/none: a/an stays a/an; bare plurals/mass stay bare; definites remain definite.
7) Keep all named entities, numerals, negation, modals, quantifier scope, and PP complements unchanged.
8) Preserve truth conditions.
9) If a well-formed Wh-cleft cannot be produced without paraphrasing or content change, set not_applicable = true with a one phrase reason.

Output format (SINGLE JSON only, no extra text):
{schema}

{fewshots_text}

Base statement: {base}
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
    variants = obj.get("variants")
    
    if isinstance(variants, dict) and not variants.get("not_applicable", False) and "text" in variants:
        return str(variants["text"]).strip()
    return None

def replace_suffix(id_str: str, new_suffix: str) -> str:
    # replace the 7 digits with the new suffix
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

def run():
    fewshots_text = render_fewshots_block(BUILTIN_FEWSHOTS)

    df = pd.read_csv(input_csv_dir)
    assert list(df.columns[:2]) == ["ID", "statement"], "CSV 前两列必须是 ID, statement"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    out_rows = []
    success_count = 0
    for i, row in df.iterrows():
        base_id = str(row["ID"])
        base_stmt = str(row["statement"])

        user_prompt = build_user_prompt(base_stmt, fewshots_text)
        raw = chat_complete(model, tokenizer, SYSTEM_PROMPT, user_prompt)
        obj = extract_first_json(raw)
        variant = pick_variant_text(obj)
        if not variant:
            print(raw)
            print(f"[warn] Row {i} JSON/variant failed, fallback to base.", file=sys.stderr)
            variant = None
        else:
            print(raw)
            print(f"Row {i} JSON/variant succeed.")
            success_count += 1

        new_id = replace_suffix(base_id, "0001000")
        out_rows.append({"ID": new_id, "statement": variant})

    pd.DataFrame(out_rows, columns=["ID", "statement"]).to_csv(output_csv_dir, index=False, encoding="utf-8")
    print(f"{success_count} / 239 rows have corresponding variants.")
    print(f"[done] Wrote: {output_csv_dir}")

if __name__ == "__main__":
    run()
