import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

input_csv_dir  = "data/original_statements.csv"
output_csv_dir = "data/SVC_variants_2.csv"

model_name = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter. "
    "Your only job is to transform the base statement into a Support Verb Construction (SVC): "
    "[SUPPORT_VERB] + [DEVERBAL_NOUN] (+ minimal required preposition) (+ original complements). "
    "Generate in English only. "
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "variant": "The state should make provision for stronger financial support to unemployed workers.",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "variant": "The EU should rigorously impose punishment on Member States that violate the EU deficit rules.",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "variant": "Bank and stock market gains should be given heavier taxation.",
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
        lines.append(f"- Base: {s['base']}\n  - SVC variant: {s['variant']}")
    return "\n".join(lines)

def build_user_prompt(base: str, fewshots_text: str) -> str:
    schema = """{
  "base": "<copy the base exactly>",
  "variants": {
      "text": "...",
      "not_applicable": false,
  }
}"""
    return f"""Task: Convert the base statement into an Support Verb Construction (SVC).

Hard constraints:
1) Make only the SVC substitution: [VERB] -> [SUPPORT_VERB] + [DEVERBAL_NOUN] (+ minimal required preposition) (+ original complements). Do not remove content. Do not make other paraphrasing.
2) If the verb in the base statement is modified by an adverb, project it as an adjective inside the SVC span if possible.
3) Keep all named entities, numerals, negation, modals, and quantifier scope unchanged.
4) Preserve complements by mapping them to the nominal head in a natural way; do not drop or invent content.
5) Aside from the SVC span and any required preposition, keep the rest of the wording identical. Preserve truth conditions.
6) If no SVC exists for the predicate, or if the base is already an SVC, set not_applicable=true.

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
    assert list(df.columns[:2]) == ["ID", "statement"], "First two columns of CSV have to be ID and statement"

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
