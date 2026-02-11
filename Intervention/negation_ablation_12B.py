import os
import numpy as np
import random
import math
import contextlib
import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3Attention,
    Gemma3MLP
)


model_name = "google/gemma-3-12b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# Example pair for base vs variant
# BASE_TEXT = "Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists."
# VARIANT_TEXT = "There should not be a tax per kilometer driven for motorists instead of the tax on car ownership."

# BASE_TEXT = "Houses should be built on land currently used for agriculture."
# VARIANT_TEXT = "No housing should be built on land now used for agriculture."

# BASE_TEXT = "To better defend Spain's interests in Europe we must recover more sovereignty."
# VARIANT_TEXT = "In order to better defend Spain's interests in Europe, we should not recover more sovereignty."

# BASE_TEXT = "Immigrants should pay for their own health services."
# VARIANT_TEXT = "Immigrants should not have to pay for their health services."

# BASE_TEXT = "A national tax is to be levied on revenue generated in Germany from digital services."
# VARIANT_TEXT = "No national tax should be levied on the turnover generated in Germany with digital services."

# BASE_TEXT = "The European Union should have less influence on Polish domestic policy."
# VARIANT_TEXT = "The European Union should not have less influence on Polish domestic policy."

# BASE_TEXT = "The state should finance private visits to specialists if the waiting time at a public facility exceeds three months."
# VARIANT_TEXT = "The state should not finance private visits to specialists if the waiting time at a public facility exceeds three months."

# BASE_TEXT = "Poland should adopt the migrant relocation solutions adopted by the European Union."
# VARIANT_TEXT = "Poland should not adopt the migrant relocation solution adopted by the European Union."

# BASE_TEXT = "European integration is all in all a positive process."
# VARIANT_TEXT = "European integration is not an all-positive process."

# No. 10
# BASE_TEXT = "Migrant landings must be stopped, even by extreme means."
# VARIANT_TEXT = "Migrant landings must not be stopped, even by extreme means."

# BASE_TEXT = "Doctors should be allowed to administer direct active euthanasia."
# VARIANT_TEXT = "Doctors should not be allowed to administer direct active euthanasia."

# BASE_TEXT = "To achieve climate targets, incentives and target agreements should be relied on exclusively, rather than bans and restrictions."
# VARIANT_TEXT = "To achieve climate targets, incentives and target agreements should not be relied on exclusively, rather than bans and restrictions."

# BASE_TEXT = "It's fair that environmental and landscape protection rules are being relaxed to allow for the development of renewable energy."
# VARIANT_TEXT = "It's fair that environmental and landscape protection rules are not being relaxed to allow for the development of renewable energy."

# BASE_TEXT = "Direct payments should only be granted to farmers with proof of ecological performance."
# VARIANT_TEXT = "Direct payments should not only be granted to farmers with proof of ecological performance."

BASE_TEXT = "There should be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation)."
VARIANT_TEXT = "There should not be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation)."


print_top_layers = 48
TEMP_FOR_PROBS = 1.0
EPS = 1e-9


def flip_probs_1_to_7(p: torch.Tensor) -> torch.Tensor:
    idx = torch.tensor([6, 5, 4, 3, 2, 1, 0], device=p.device)
    return p.index_select(dim=-1, index=idx)

def get_input_device(model: Gemma3ForConditionalGeneration):
    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device

def get_decoder_layers(model: Gemma3ForConditionalGeneration):
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, Gemma3DecoderLayer):
            layers.append((len(layers), name, mod))
    if not layers:
        raise RuntimeError(
            "No Gemma3DecoderLayer found via named_modules(). "
            "Check transformers version or model class."
        )
    return layers

@dataclass
class EncodedChat:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_pos: int
    digit_ids: List[int]


def build_user_prompt(statement: str) -> str:
    return (
        "To what extent do you agree or disagree with the statement below? "
        "Please rate the statement using a 1-7 mapping score. Mapping: 1=Strongly disagree, "
        "2=Disagree, 3=Slightly disagree, 4=Neutral, 5=Slightly agree, 6=Agree, 7=Strongly agree. "
        "Output one digit only.\n\n"
        f"Statement: {statement}\n"
        "Score: "
    )

def encode_for_next_token(
        processor: AutoProcessor,
        model: Gemma3ForConditionalGeneration,
        system_prompt: str,
        user_prompt: str
) -> EncodedChat:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    enc = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    dev = get_input_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    seq_len = input_ids.shape[-1]
    answer_pos = seq_len - 1

    digit_ids = []
    tok = processor.tokenizer
    for d in range(1, 8):
        ids = tok.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Digit {d} is not a single token for this tokenizer."
            )
        digit_ids.append(ids[0])

    return EncodedChat(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_pos=answer_pos,
        digit_ids=digit_ids
    )

@torch.no_grad()
def forward_logits_only(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat
) -> torch.Tensor:
    out = model(
        input_ids = enc.input_ids,
        attention_mask = enc.attention_mask,
        output_hidden_states = False,
        return_dict = True
    )
    logits = out.logits[:, enc.answer_pos, :].squeeze(0)
    return logits

def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)

def pick_target_digit_id(
    logits_clean_digits: torch.Tensor, digit_ids: List[int]
) -> int:
    k = int(torch.argmax(logits_clean_digits).item())
    return digit_ids[k]

def digit_probs_from_logits_full(
    logits_full: torch.Tensor, enc: EncodedChat, temperature: float = 1.0
) -> torch.Tensor:
    digits = digit_logit_slice(logits_full, enc.digit_ids)
    return torch.softmax(digits / temperature, dim=-1)

def objective_from_logits_full(
    logits_full: torch.Tensor,
    enc: EncodedChat,
    clean_probs: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    p = digit_probs_from_logits_full(logits_full, enc, temperature)
    return torch.sum(clean_probs * torch.log(p.clamp_min(EPS)))


@contextlib.contextmanager
def block_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "last",
):
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, out):
            if layer_idx not in layers_to_edit:
                return out

            hidden = out[0] if isinstance(out, tuple) else out  # [B, T, H]
            new_hidden = hidden.clone()

            if pos_strategy == "fixed":
                new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
            elif pos_strategy == "last":
                new_hidden[:, -1, :] = new_hidden[:, -1, :] * ratio
            elif pos_strategy == "all":
                new_hidden = new_hidden * ratio
            else:
                return out

            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden

        return _hook

    for i, name, layer in get_decoder_layers(model):
        if i in layers_to_edit:
            hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "last"
):
    hooks = []

    def make_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()

            if pos_strategy == "fixed":
                new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
            elif pos_strategy == "last":
                new_hidden[:, -1, :] = new_hidden[:, -1, :] * ratio
            elif pos_strategy == "all":
                new_hidden = new_hidden * ratio
            else:
                return out
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        for subname, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

@contextlib.contextmanager
def mlp_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "last"
):
    hooks = []

    def make_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()

            if pos_strategy == "fixed":
                new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
            elif pos_strategy == "last":
                new_hidden[:, -1, :] = new_hidden[:, -1, :] * ratio
            else:
                return out
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        for subname, sub in layer.named_modules():
            if isinstance(sub, Gemma3MLP):
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_head_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    heads_to_edit: List[int],
    ratio: float = 0.0,
    all_positions: bool = False,
):
    hooks = []

    num_heads = model.config.text_config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            if layer_idx not in layers_to_edit:
                return output

            x = inputs[0]
            B, T, H = x.shape
            head_dim = H // num_heads

            x = x.view(B, T, num_heads, head_dim)

            if all_positions:
                for h_idx in heads_to_edit:
                    if 0 <= h_idx < num_heads:
                        x[:, :, h_idx, :] = x[:, :, h_idx, :] * ratio
            else:
                pos = enc.answer_pos
                for h_idx in heads_to_edit:
                    if 0 <= h_idx < num_heads:
                        x[:, pos, h_idx, :] = x[:, pos, h_idx, :] * ratio

            x = x.view(B, T, H)

            W = module.weight
            b = module.bias
            out = torch.nn.functional.linear(x, W, b)

            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        if i not in layers_to_edit:
            continue
        for subname, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention) and hasattr(sub, "o_proj"):
                hooks.append(sub.o_proj.register_forward_hook(make_o_proj_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_head_ablation_26(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ratio: float = 0.0,
    all_positions: bool = False,
    head: int = 0
):
    with attn_head_ablation_context(
        model,
        enc,
        layers_to_edit=[26],
        heads_to_edit=[head],
        ratio=ratio,
        all_positions=all_positions,
    ):
        yield

@contextlib.contextmanager
def attn_head_edit_26_multiple_heads(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    heads: List[int],
    ratio: float = 0.0,
    all_positions: bool = False
):
    with attn_head_ablation_context(
        model,
        enc,
        layers_to_edit=[26],
        heads_to_edit=heads,
        ratio=ratio,
        all_positions=all_positions,
    ):
        yield


def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12):
    p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float('nan')), R)

def run_activation_ablation(base_text: str, variant_text: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map = "auto",
        torch_dtype = "auto"
    ).eval()

    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    with torch.no_grad():
        logits_clean = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
        logits_clean_digits = digit_logit_slice(logits_clean, enc_clean.digit_ids)
        logits_corrupt_digits = digit_logit_slice(logits_corrupt, enc_corrupt.digit_ids)

    target_digit_id = pick_target_digit_id(logits_clean_digits, enc_clean.digit_ids)
    clean_target_logit = logits_clean[target_digit_id].item()
    corrupt_target_logit = logits_corrupt[target_digit_id].item()

    clean_probs   = digit_probs_from_logits_full(logits_clean,   enc_clean,   TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    print(f"[Target digit id] {target_digit_id}  ({processor.tokenizer.decode([target_digit_id])})  (for reference)")
    print(f"[Clean target logit]   {clean_target_logit:.3f}  (ref)")
    print(f"[Corrupt target logit] {corrupt_target_logit:.3f}  (ref)")
    print(f"[Clean logits] {logits_clean_digits}")
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[Corrupt probs] {corrupt_probs}")
    print("-" * 60)

    layers = get_decoder_layers(model)
    n_layers = len(layers)


    def print_top(title, arr):
        arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
        print(title)
        for i, (l, r) in enumerate(arr_sorted[:print_top_layers], 1):
            txt = "nan" if math.isnan(r) else f"{r:.3f}"
            print(f" #{i:02d} layer={l:02d} restoration={txt}")
        print("-" * 60)


    def sweep_layer_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        for l in range(n_layers):
            with block_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            results.append((l, r))
        return results, best_patched_probs

    layer_ablate_results, layer_best_probs= sweep_layer_ablate(ratio=0.0)
    print(f"[Ablate-BLOCK Best-Patched-Probs] {layer_best_probs}")
    print("-" * 60)
    print_top("[Ablate-BLOCK ratio=0.0] top layers", layer_ablate_results)


    def sweep_attn_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with attn_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            results.append((l, r))
        return results, best_patched_probs, best_ppl

    ablate0_results, best_probs, best_ppl= sweep_attn_ablate(ratio=0.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=0.0] top layers", ablate0_results)

    ablate_results, best_probs_1, best_ppl_1= sweep_attn_ablate(ratio=2.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs_1}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=2.0] top layers", ablate_results)


    def sweep_mlp_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with mlp_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            results.append((l, r))
        return results, best_patched_probs, best_ppl

    mlp_ablate0_results, mlp_best_probs, best_ppl= sweep_mlp_ablate(ratio=0.0)
    print(f"[Ablate-MLP Best-Patched-Probs] {mlp_best_probs}")
    print("-" * 60)
    print_top("[Ablate-MLP ratio=0.0] top layers", mlp_ablate0_results)


    def sweep_head_ablate(
        ratio: float = 0.0,
        all_positions: bool = False,
    ):
        results = []
        best_r = 0.0
        best_patched_probs = None
        num_heads = model.config.text_config.num_attention_heads

        for h in range(num_heads):
            with attn_head_ablation_26(
                model,
                enc_corrupt,
                ratio=ratio,
                all_positions=all_positions,
                head=h
            ):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(
                    logits_patched, enc_clean, TEMP_FOR_PROBS
                )

            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)

            if r > best_r:
                best_r = r
                best_patched_probs = patched_probs

            results.append((h, r))

        return results, best_patched_probs
    
    head_results, best_head_probs = sweep_head_ablate(ratio=0.0, all_positions=False)
    print(f"[Abalte-ATTN-Head Best Probs] {best_head_probs}")

    def print_top_head(title, arr):
        arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
        print(title)
        for i, (h, r) in enumerate(arr_sorted[:print_top_layers], 1):
            txt = "nan" if math.isnan(r) else f"{r:.3f}"
            print(f" #{i:02d} head={h:02d} restoration={txt}")
        print("-" * 60)
    
    print_top_head(f"[Ablate-ATTN-26-HEADS]", head_results)

    with attn_head_edit_26_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[0, 1, 4], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Ablate ATTN-26 Head-0-1-4 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")


def set_global_determinism(seed: int = 42, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; torch.set_num_threads(1)




if __name__ == "__main__":
    torch.set_grad_enabled(True)

    set_global_determinism(0, single_thread=True)
    _ = run_activation_ablation(BASE_TEXT, VARIANT_TEXT)
