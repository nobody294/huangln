import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CSV file paths
input_csv_dir  = "data/original_statements.csv"
output_csv_dir = "data/LVC_variants.csv"

model_name = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter. "
    "Your only job is to transform the base statement into a Light-Verb Construction (LVC): "
    "[LIGHT_VERB] + [DEVERBAL_NOUN] (+ minimal required preposition) (+ original complements). "
    "Truth conditions must be preserved. Do not remove content. Do not paraphrase. "
    "Preserve scope of negation, modals, quantifiers, tense/aspect, numbers, named entities, and PPs (time/place). "
    "Generate in English only. "
    "Output a single JSON object exactly matching the schema."
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "variant": "null",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "variant": "The EU should rigorously impose punishment on Member States that violate the EU deficit rules.",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "variant": "Heavier taxation should be imposed on bank and stock market gains.",
    },
    {
        "base": "In European Parliament elections, EU citizens should be allowed to cast a vote for a party or candidate from any other Member State.",
        "variant": "In European Parliament elections, EU citizens should be given permission to cast a vote for a party or candidate from any other Member State.",
    },
    {
        "base": "The legalisation of same sex marriages is a good thing.",
        "variant": "null",
    },
    {
        "base": "The legalisation of the personal use of soft drugs is to be welcomed.",
        "variant": "null",
    },
]

def render_fewshots_block(shots):
    lines = ["Few-shot exemplars (follow style strictly):"]
    for s in shots:
        lines.append(f"- Base: {s['base']}\n  - LVC variant: {s['variant']}")
    return "\n".join(lines)

def build_user_prompt(base: str, fewshots_text: str) -> str:
    schema = """{
  "base": "<copy the base exactly>",
  "variants": {
      "text": "...",
      "not_applicable": false,
      "reason": null
  }
}"""
    return f"""Task: Convert the base statement into an Light-Verb Construction (LVC).

Hard constraints (follow strictly):
1) Make only the LVC substitution: [VERB] -> [LIGHT_VERB] + [DEVERBAL_NOUN] (+ minimal required preposition) (+ original complements). Do not remove content. Do not make other paraphrasing.
2) Keep all named entities, numerals, negation, modals, quantifier scope, and PP complements unchanged.
3) Preserve complements by mapping them to the nominal head in a natural way; do not drop or invent content.
4) If no LVC exists for the predicate, if the base is already an LVC, or if the base is non-eventive/copular, set not_applicable=true and give a brief reason (one phrase).
5) Aside from the LVC span and any required preposition, keep the rest of the wording identical.

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
