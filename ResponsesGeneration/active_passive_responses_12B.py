import re, json, sys
import pandas as pd
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

input_csv_dir = "data/active_passive_variants.csv"
output_csv_dir = "data/active_passive_responses_12B.csv"

model_name = "google/gemma-3-12b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

def build_user_prompt_for_scoring(statement: str) -> str:
    return (
        "To what extent do you agree or disagree with the statement below? "
        "Please rate the statement using a 1-7 mapping score. Mapping: 1=Strongly disagree, 2=Disagree, 3=Slightly disagree, 4=Neutral, 5=Slightly agree, 6=Agree, 7=Strongly agree. "
        "Output one digit only.\n\n"
        f"Statement: {statement}\n"
        "Score: "
    )

SCORE_RE_STRICT = re.compile(r"Score\s*:\s*([1-7])\b", re.IGNORECASE)
SCORE_RE_LOOSE = re.compile(r"\b([1-7])\b")

def clamp_score(x):
    try:
        xi = int(x)
        return xi if 1 <= xi <= 7 else None
    except Exception:
        return None

def extract_first_score(s: str):
    m = SCORE_RE_STRICT.search(s)
    if not m:
        m = SCORE_RE_LOOSE.search(s)
    return clamp_score(m.group(1)) if m else None

def generate_30_json_responses(model, processor, system_prompt, user_prompt,
                               temperature=0.8, top_p=0.95, max_new_tokens=4, seed=42):
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

    gen_only = gen_ids[:, input_len:]
    decoded = processor.batch_decode(gen_only, skip_special_tokens=True)
    return decoded

def run():
    df = pd.read_csv(input_csv_dir)
    assert list(df.columns[:2]) == ["ID", "statement"], "First two columns of CSV have to be ID and statement"

    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    ).eval()

    out_rows = []
    total_ok = 0
    for i, row in df.iterrows():
        id = str(row["ID"])

        stmt = row["statement"]
        if not (isinstance(stmt, str) and stmt.strip()):
            sys.stdout.write(f"[Row {i+1}] ID={id} -> skipped (empty statement)\n")
            sys.stdout.flush()
            continue

        user_prompt = build_user_prompt_for_scoring(str(stmt))
        raw_list = generate_30_json_responses(
            model, processor, SYSTEM_PROMPT, user_prompt,
            temperature=0.8, top_p=0.95, max_new_tokens=4, seed=123 + i
        )

        ok_count = 0
        for raw in raw_list:
            score = extract_first_score(raw)
            if score is None:
                continue

            out_rows.append({"ID": id, "score": score})
            ok_count += 1

        total_ok += ok_count
        sys.stdout.write(f"[Row {i+1}] ID={id} -> parsed {ok_count}/30 JSONs\n")
        sys.stdout.flush()

    pd.DataFrame(out_rows, columns=["ID", "score"]).to_csv(
        output_csv_dir, index=False, encoding="utf-8"
    )
    print(f"[done] Collected {total_ok} responses across {len(df)} statements.")
    print(f"[done] Wrote: {output_csv_dir}")

if __name__ == "__main__":
    run()
