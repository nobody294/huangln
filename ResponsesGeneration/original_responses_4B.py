import re, json, sys
import pandas as pd
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

INPUT_CSV = "data/original_statements.csv"
OUTPUT_CSV = "data/original_responses_4B.csv"

model_name = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 7-point Likert scale."
)

def build_user_prompt_for_scoring(statement: str) -> str:
    schema = """{
    "score": 1|2|3|4|5|6|7
}"""
    return f"""To what extent do you agree or disagree with the statement below? Please rate the statement using a 1-7 mapping score. Mapping: 1=Strongly disagree, 2=Disagree, 3=Slightly disagree, 4=Neutral, 5=Slightly agree, 6=Agree, 7=Strongly agree. Output a single JSON object exactly matching the schema.

    Output format (single JSON only, no extra text):
    {schema}

    Statement: {statement}
"""

# --- JSON helpers ----------------------------------------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(s: str):
    m = JSON_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def clamp_score(x):
    try:
        xi = int(x)
        return xi if 1 <= xi <= 7 else None
    except Exception:
        return None

# --- Generation with Gemma-3 ----------------------------------------------

def generate_30_json_responses(model, processor, system_prompt, user_prompt,
                               temperature=0.8, top_p=0.95, max_new_tokens=50, seed=42):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    torch.manual_seed(seed)
    gen_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=30,
        pad_token_id=(model.generation_config.pad_token_id
                      if model.generation_config.pad_token_id is not None
                      else model.generation_config.eos_token_id)
    )

    # 只取生成段
    gen_only = gen_ids[:, input_len:]
    decoded = processor.batch_decode(gen_only, skip_special_tokens=True)
    return decoded

# --- Main -----------------------------------------------------------------

def run():
    df = pd.read_csv(INPUT_CSV)
    assert list(df.columns[:2]) == ["ID", "statement"], "CSV 前两列必须是 ID, statement"

    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",          # 自动放置
        torch_dtype="auto"          # 自动选择 dtype（有 bfloat16 更佳）
    ).eval()

    # 若无 pad_token_id，回退到 eos_token_id，避免 generate 警告
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    out_rows = []
    total_ok = 0
    for i, row in df.iterrows():
        id = str(row["ID"])
        stmt = str(row["statement"])

        user_prompt = build_user_prompt_for_scoring(stmt)
        raw_list = generate_30_json_responses(
            model, processor, SYSTEM_PROMPT, user_prompt,
            temperature=0.8, top_p=0.95, max_new_tokens=50, seed=123 + i
        )

        ok_count = 0
        for raw in raw_list:
            obj = extract_first_json(raw)
            if not isinstance(obj, dict): 
                continue

            score = clamp_score(obj.get("score"))
            if score is None:
                continue

            out_rows.append({"ID": id, "score": score})
            ok_count += 1

        total_ok += ok_count
        sys.stdout.write(f"[Row {i}] ID={id} -> parsed {ok_count}/30 JSONs\n")
        sys.stdout.flush()

    pd.DataFrame(out_rows, columns=["ID", "score"]).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8"
    )
    print(f"[done] Collected {total_ok} responses across {len(df)} statements.")
    print(f"[done] Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
