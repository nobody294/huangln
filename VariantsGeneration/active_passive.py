import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CSV file paths
input_csv_dir  = "data/original_statements.csv"
output_csv_dir = "data/active_passive_variants.csv"

model_name = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are a controlled text rewriter. "
    "Your only job is to convert the base statement between active and passive voice, "
    "preserving truth-conditional meaning, named entities, numbers, scope, modality, tense, aspect, and negation. "
    "Do not add or remove qualifiers. Generate in English only. "
    "Output a single JSON object exactly matching the schema."
)

BUILTIN_FEWSHOTS = [
    {
        "base": "The state should provide stronger financial support to unemployed workers.",
        "variant": "Stronger financial support should be provided to unemployed workers by the state.",
        "direction": "active_to_passive",
    },
    {
        "base": "The EU should rigorously punish Member States that violate the EU deficit rules.",
        "variant": "Member States that violate the EU deficit rules should be rigorously punished by the EU.",
        "direction": "active_to_passive",
    },
    {
        "base": "Bank and stock market gains should be taxed more heavily.",
        "variant": "The government should tax bank and stock market gains more heavily.",
        "direction": "passive_to_active",
    },
]

def render_fewshots_block(shots):
    lines = ["Few-shot exemplars (follow style strictly):"]
    for s in shots:
        dir_tag = s.get("direction", "unknown")
        tag = "active->passive" if dir_tag == "active_to_passive" else (
              "passive->active" if dir_tag == "passive_to_active" else "voice conversion")
        lines.append(f"- Base: {s['base']}\n  - {tag}: {s['variant']}")
    return "\n".join(lines)

def build_user_prompt(base: str, fewshots_text: str) -> str:
    schema = """{
  "base": "<copy the base exactly>",
  "direction": "active_to_passive" | "passive_to_active" | "unknown",
  "variants": [
    {
      "text": "...",
      "edit_ops": ["voice: active->passive" | "voice: passive->active"],
      "not_applicable": false,
      "reason": none
    }
  ],
  "self_check": {
    "arguments_preserved": true,
    "modals_tense_aspect_preserved": true,
    "negation_scope_preserved": true
  }
}"""
    return f"""Task: Convert the base statement between active and passive voice.

Hard constraints:
1) Do exactly and only a voice transformation (Active<->Passive). Preserve all arguments, named entities, numbers, tense/aspect, modals, quantifiers, negation scope, and PPs.
2) If an agent exists, keep it (use a by-phrase in passive).
3) If voice transformation is inapplicable (e.g., copular predicates without a transitive verb), set not_applicable=true and reason=a one-phrase reason.
4) Keep truth-conditional meaning intact. No paraphrasing beyond voice change.

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
    variants = obj.get("variants") or []
    for v in variants:
        if isinstance(v, dict) and not v.get("not_applicable", False) and "text" in v:
            return str(v["text"]).strip()
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
                  max_new_tokens=512, temperature=0.3, top_p=0.9):
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
    output_ids = out[0][len(inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
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
        else:
            print(f"Row {i} JSON/variant succeed.")

        new_id = replace_suffix(base_id, "0001000")
        out_rows.append({"ID": new_id, "statement": variant})

    pd.DataFrame(out_rows, columns=["ID", "statement"]).to_csv(output_csv_dir, index=False, encoding="utf-8")
    print(f"[done] Wrote: {output_csv_dir}")

if __name__ == "__main__":
    run()
