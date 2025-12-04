import json
from typing import NamedTuple
import os
import numpy as np
import random
import re, csv
from collections import defaultdict
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
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

# ---------------------------
# Config
# ---------------------------

model_name = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

MMLU_SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. "
    "Answer correctly and output exactly one capital letter: A, B, C, or D."
)

# Example pair for base vs variant (you can change these)
BASE_TEXT = "The government should abolish the ban on face-covering clothing."
VARIANT_TEXT = "It is the ban on face-covering clothing that the government should abolish."

# BASE_TEXT = "Houses should be built on land currently used for agriculture."
# VARIANT_TEXT = "It is on land currently used for agriculture that Houses should be built."

# BASE_TEXT = "The Netherlands should introduce an additional flight tax for short-distance flights."
# VARIANT_TEXT = "It is an additional flight tax for short-distance flights that the Netherlands should introduce."

# BASE_TEXT = "An increase in minimum wages should no longer automatically result in an increase in welfare benefits."
# VARIANT_TEXT = "It is an increase in welfare benefits that an increase in minimum wages should no longer automatically result in."

# BASE_TEXT = "People should always have the choice of whether to wear a face mask."
# VARIANT_TEXT = "It is the choice of whether to wear a face mask that people should always have."

# BASE_TEXT = "The future Spanish government should increase irrigated agricultural areas by means of large water transfers."
# VARIANT_TEXT = "It is irrigated agricultural areas that the future Spanish government should increase by means of large water transfers."

# BASE_TEXT = "Spain should be more tolerant with illegal migration."
# VARIANT_TEXT = "It is illegal migration that Spain should be more tolerant with."

# BASE_TEXT = "Donations from companies to political parties should continue to be permitted."
# VARIANT_TEXT = "It is donations from companies to political parties that should continue to be permitted."

# BASE_TEXT = "The federal government is to be given more responsibilities in school policy."
# VARIANT_TEXT = "It is more responsibilities in school policy that the federal government is to be given."

# BASE_TEXT = "Chinese companies should not be allowed to receive contracts for the expansion of the communications infrastructure in Germany."
# VARIANT_TEXT = "It is contracts for the expansion of the communications infrastructure in Germany that Chinese companies should not be allowed to receive."

# BASE_TEXT = "A tax is to be levied again on high assets."
# VARIANT_TEXT = "It is high assets that a tax is to be levied again on."

# BASE_TEXT = "Facial recognition software should be allowed to be used for video surveillance in public places."
# VARIANT_TEXT = "It is video surveillance in public places that facial recognition software should be allowed to be used for."

# BASE_TEXT = "Married couples without children should continue to receive tax breaks."
# VARIANT_TEXT = "It is tax breaks that Married couples without children should continue to receive."

# BASE_TEXT = "Air traffic is to be taxed more heavily."
# VARIANT_TEXT = "It is air traffic that is to be taxed more heavily."

# BASE_TEXT = "The share of defense spending in Poland's GDP should be further increased."
# VARIANT_TEXT = "It is the share of defense spending in Poland's GDP that should be further increased."

# BASE_TEXT = "Hungary should decide by referendum whether to remain part of the EU."
# VARIANT_TEXT = "It is by referendum that Hungary should decide whether to remain part of the EU."

# BASE_TEXT = "Hungary should join the European Public Prosecutor's Office."
# VARIANT_TEXT = "It is the European Public Prosecutor's Office that Hungary should join."

# BASE_TEXT = "Only men and women should be allowed to marry."
# VARIANT_TEXT = "It is only men and women that should be allowed to marry."

# BASE_TEXT = "Parties should strive for a closer ratio of men to women when drawing up lists."
# VARIANT_TEXT = "It is a closer ratio of men to women that Parties should strive for when drawing up lists."

# BASE_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation."
# VARIANT_TEXT = "It is inflation that a price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight."

# BASE_TEXT = "Italy should get out of the Eurozone."
# VARIANT_TEXT = "It is the Eurozone that Italy should get out of."

# BASE_TEXT = ""
# VARIANT_TEXT = ""

# BASE_TEXT = ""
# VARIANT_TEXT = ""

topk_attr = 6          # how many top layers to print/consider in diagnostics
print_top_layers = 20  # how many top layers to print
TEMP_FOR_PROBS = 1.0
EPS = 1e-9

# ---------------------------
# Utilities / Model Introspection
# ---------------------------

# === Multi-layer feature extraction & fusion (H-dim) ===
from typing import Union, Sequence

def _layer_key(layers: Union[int, Sequence[int]]) -> str:
    if isinstance(layers, int):
        return f"L{layers}"
    return "L" + "+".join(str(int(l)) for l in layers)

@torch.no_grad()
def _extract_hidden_for_text(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    text: str,
    layers: Sequence[int],
    input_mode: str = "chat",  # "chat" 或 "raw"
    system_prompt: str = "",
    raw_max_len: int = 1024,
    pool: str = "mean",        # 仍用 mean pool
) -> List[torch.Tensor]:
    """
    返回: [h_l1, h_l2, ...]，每个 h_li 的形状为 [H]，均为 CPU float32。
    """
    dev = get_input_device(model)
    if input_mode == "chat":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": build_user_prompt(text)}]},
        ]
        enc = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True
        )
    else:
        tok = processor.tokenizer
        enc = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=raw_max_len)
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    feats = []
    for li in layers:
        hs = out.hidden_states[li]          # [1, seq, H]
        feat = hs.mean(dim=1).squeeze(0)    # [H]
        feats.append(feat.to(torch.float32).cpu())
    return feats

def _fuse_hdim(
    feats: Sequence[torch.Tensor],
    mode: str = "sum",          # "sum" | "mean" | "fixed"
    weights: Optional[Sequence[float]] = None
) -> torch.Tensor:
    """
    返回融合后的 [H] 张量（CPU float32）。
    """
    assert len(feats) >= 2, "需要至少两层特征"
    H = feats[0].numel()
    for f in feats:
        assert f.numel() == H, "所有层的 hidden_size 必须一致"

    if mode == "sum":
        x = torch.stack(feats, dim=0).sum(dim=0)
    elif mode == "mean":
        x = torch.stack(feats, dim=0).mean(dim=0)
    elif mode == "fixed":
        assert weights is not None and len(weights) == len(feats)
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / (w.sum() + 1e-12)
        x = torch.stack(feats, dim=0)       # [L,H]
        x = (w.view(-1,1) * x).sum(dim=0)   # [H]
    else:
        raise ValueError(f"Unknown fuse mode: {mode}")
    return x.contiguous()


def get_input_device(model: Gemma3ForConditionalGeneration):
    # More robust than model.device under device_map="auto"
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
    # print(f"{len(layers)}")
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

# ---------------------------
# Attribution & Caches (diagnostics)
# ---------------------------

def attribution_scores_first_order(
        model: Gemma3ForConditionalGeneration,
        enc_clean: EncodedChat,
        enc_corrupt: EncodedChat,
        clean_probs: Optional[torch.Tensor]
):
    clean_cache = collect_clean_cache(model, enc_clean)

    h_corrupt_pos: Dict[int, torch.Tensor] = {}
    grad_pos: Dict[int, torch.Tensor] = {}
    fwd_hooks, bwd_hooks = [], []

    def make_fwd_hook(layer_idx):
        def _fwd(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            v = hidden[:, enc_corrupt.answer_pos, :].detach().squeeze(0)
            h_corrupt_pos[layer_idx] = v
            return out
        return _fwd

    def make_bwd_hook(layer_idx):
        def _bwd(module, grad_input, grad_output):
            g = grad_output[0][:, enc_corrupt.answer_pos, :].detach().squeeze(0)
            grad_pos[layer_idx] = g
        return _bwd

    for i, name, layer in get_decoder_layers(model):
        fwd_hooks.append(layer.register_forward_hook(make_fwd_hook(i)))
        bwd_hooks.append(layer.register_full_backward_hook(make_bwd_hook(i)))

    out = model(
        input_ids=enc_corrupt.input_ids,
        attention_mask=enc_corrupt.attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    logits_corrupt = out.logits[:, enc_corrupt.answer_pos, :].squeeze(0)

    obj = objective_from_logits_full(
        logits_corrupt, enc_corrupt, clean_probs, TEMP_FOR_PROBS
    )

    model.zero_grad(set_to_none=True)
    obj.backward(retain_graph=False)

    for h in fwd_hooks + bwd_hooks:
        h.remove()

    scores = []
    device = logits_corrupt.device
    layer_ids = sorted(set(clean_cache.block_out.keys()) &
                    set(h_corrupt_pos.keys()) &
                    set(grad_pos.keys()))
    for l in layer_ids:
        hc = clean_cache.block_out[l].to(device)
        hr = h_corrupt_pos[l].to(device)
        g  = grad_pos[l].to(device)
        s = torch.dot((hc - hr).float(), g.float()).item()
        scores.append((l, s))
    scores_sorted = sorted(scores, key=lambda x: abs(x[1]), reverse=True)
    return scores_sorted


class CleanCache:
    def __init__(self):
        self.block_out: Dict[int, torch.Tensor] = {}
        self.attn_out: Dict[int, torch.Tensor] = {}
        self.mlp_out: Dict[int, torch.Tensor] = {}

    def to_device_like(self, ref: torch.Tensor):
        for d in (self.block_out, self.attn_out, self.mlp_out):
            for k, v in d.items():
                if v.device != ref.device:
                    d[k] = v.to(ref.device)


def collect_clean_cache(
        model: Gemma3ForConditionalGeneration,
        enc_clean: EncodedChat
) -> CleanCache:
    cache = CleanCache()
    hooks = []

    def layer_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.block_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def attn_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.attn_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def mlp_hook(layer_idx):
        def _hook(module, input, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.mlp_out[layer_idx] = vec.cpu()
            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_hook(i)))
        for subname, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Gemma3MLP):
                hooks.append(sub.register_forward_hook(mlp_hook(i)))

    with torch.no_grad():
        _ = model(
            input_ids = enc_clean.input_ids,
            attention_mask = enc_clean.attention_mask,
            output_hidden_states = False,
            return_dict = True
        )

    for h in hooks:
        h.remove()
    return cache

@contextlib.contextmanager
def patch_context(
    model: Gemma3ForConditionalGeneration,
    enc_corrupt: EncodedChat,
    cache: CleanCache,
    patch_spec: Dict[str, List[int]]
):
    hooks = []
    cache.to_device_like(enc_corrupt.input_ids)

    def replace_slice(hidden: torch.Tensor, vec: torch.Tensor):
        new_hidden = hidden.clone()
        new_hidden[:, enc_corrupt.answer_pos, :] = vec.to(hidden.dtype).to(hidden.device)
        return new_hidden

    def layer_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("block", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.block_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def attn_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("attn", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.attn_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def mlp_patch_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in patch_spec.get("mlp", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.mlp_out[layer_idx].to(hidden.device)
            new_hidden = replace_slice(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_patch_hook(i)))
        for subname, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Gemma3MLP):
                hooks.append(sub.register_forward_hook(mlp_patch_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

# def restoration_fraction(
#         logit_clean_target: float,
#         logit_corrupt_target: float,
#         logit_patched_target: float
# ) -> float:
#     denom = logit_clean_target - logit_corrupt_target
#     if abs(denom) < 1e-9:
#         return float("nan")

#     return (logit_patched_target - logit_corrupt_target) / denom


@contextlib.contextmanager
def attn_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0
):
    hooks = []

    def make_hook(layer_idx):
        def _hook(module, input, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()
            new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
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

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps=1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)

def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12):
    d0 = dist_fn(p_clean, p_corrupt)
    dp = dist_fn(p_clean, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float('nan')), R)

def run_activation_patching(base_text: str, variant_text: str):
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

    c_id = pick_target_digit_id(logits_corrupt_digits, enc_corrupt.digit_ids)

    print(f"[Target digit id] {target_digit_id}  ({processor.tokenizer.decode([target_digit_id])})")
    # print(f"[Clean target logit]   {clean_target_logit:.3f}")
    print(f"[Clean logits] {logits_clean_digits}")
    # print(f"[Corrupt target logit] {corrupt_target_logit:.3f}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[c_id] {c_id}  ({processor.tokenizer.decode([c_id])})")
    print("-" * 60)

    clean_probs   = digit_probs_from_logits_full(logits_clean,   enc_clean,   TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    obj_clean = objective_from_logits_full(
        logits_clean, enc_clean, clean_probs, TEMP_FOR_PROBS
    ).item()
    obj_corrupt = objective_from_logits_full(
        logits_corrupt, enc_corrupt, clean_probs, TEMP_FOR_PROBS
    ).item()

    print(f"[Target digit id] {target_digit_id}  ({processor.tokenizer.decode([target_digit_id])})  (for reference)")
    print(f"[Clean target logit]   {clean_target_logit:.3f}  (ref)")
    print(f"[Corrupt target logit] {corrupt_target_logit:.3f}  (ref)")
    print(f"[Clean logits] {logits_clean_digits}")
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[Corrupt probs] {corrupt_probs}")
    print("-" * 60)

    # scores_sorted = attribution_scores_first_order(
    #     model, enc_clean, enc_corrupt, clean_probs
    # )
    # print("[Attribution (first-order) — top layers]")
    # for i, (l, s) in enumerate(scores_sorted[:print_top_layers], 1):
    #     print(f" #{i:02d} layer={l:02d} approx_gain={s:+.3e}")
    # print("-" * 60)

    clean_cache = collect_clean_cache(model, enc_clean)
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    def sweep_patch(kind: str) -> List[Tuple[int, float]]:
        results = []
        for l in range(n_layers):
            spec = {"block": [], "attn": [], "mlp": []}
            spec[kind] = [l]
            with patch_context(model, enc_corrupt, clean_cache, spec):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
            obj_patched = objective_from_logits_full(
                logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
            ).item()
            # denom = obj_clean - obj_corrupt
            # if abs(denom) < 1e-9:
            #     r = float("nan")
            # else:
            #     r = (obj_patched - obj_corrupt) / denom
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, r))
        return results

    block_results = sweep_patch("block")
    attn_results = sweep_patch("attn")
    mlp_results = sweep_patch("mlp")

    def print_top(title, arr):
        arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
        print(title)
        for i, (l, r) in enumerate(arr_sorted[:print_top_layers], 1):
            txt = "nan" if math.isnan(r) else f"{r:.3f}"
            print(f" #{i:02d} layer={l:02d} restoration={txt}")
        print("-" * 60)

    print_top("[Patch - BLOCK - top layers]", block_results)
    print_top("[Patch - ATTN - top layers]", attn_results)
    print_top("[Patch - MLP - top layers]", mlp_results)

    def sweep_attn_ablate(ratio: float = 0.0) -> List[Tuple[int, float]]:
        results = []
        for l in range(n_layers):
            with attn_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
            obj_patched = objective_from_logits_full(
                logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
            ).item()
            # denom = obj_clean - obj_corrupt
            # if abs(denom) < 1e-9:
            #     r = float("nan")
            # else:
            #     r = (obj_patched - obj_corrupt) / denom
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, r))
        return results

    ablate0_results = sweep_attn_ablate(ratio=0.0)
    print_top("[Ablate-ATTN ratio=0.0] top layers", ablate0_results)

# ---------------------------
# Paper-faithful pipeline (multi-GPU safe) — CE probe + Δw + min-cos rows + gate_proj add
# ---------------------------
def set_global_determinism(seed: int = 42, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # 或 ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; torch.set_num_threads(1)
# ===========================
# CSV IO: read train_base / train_variant and pair by ID prefix with non-empty statements
# ===========================

# ID 形如: 两个小写字母_一或两位数字_七位数字
# 例如: ab_3_1234567 或 xy_12_7654321
# 我们把“前两段”（如 "ab_3"）当作配对前缀键
_ID_RE = re.compile(r'^([a-z]{2}_[0-9]{1,2})_([0-9]{7})$')

def _extract_prefix(id_str: str) -> Optional[str]:
    if not id_str:
        return None
    m = _ID_RE.match(id_str.strip())
    return m.group(1) if m else None

def _is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())

def _load_id_stmt_map(csv_path: str):
    pref2pair: Dict[str, Tuple[str, str]] = {}
    stats = {
        "rows": 0,
        "bad_id": 0,
        "empty_stmt": 0,
        "dup_prefix": 0,
    }
    seen_prefix = set()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'ID' not in reader.fieldnames or 'statement' not in reader.fieldnames:
            raise ValueError(f"{csv_path} 必须包含列: ID, statement（区分大小写）")
        for row in reader:
            stats["rows"] += 1
            id_raw = (row.get('ID') or '').strip()
            stmt   = (row.get('statement') or '').strip()

            pref = _extract_prefix(id_raw)
            if not pref:
                stats["bad_id"] += 1
                continue

            if pref in seen_prefix:
                stats["dup_prefix"] += 1
                continue
            # 只有非空 statement 才保留
            if not _is_nonempty_text(stmt):
                stats["empty_stmt"] += 1
                # 不加入映射，这样后面配对时自然会被滤掉
                seen_prefix.add(pref)  # 同一前缀后续再遇到也算重复，避免歧义
                continue

            pref2pair[pref] = (id_raw, stmt)
            seen_prefix.add(pref)

    return pref2pair, stats

def _load_flip_prefix_set(csv_path: str):
    flip_prefixes = set()
    stats = {
        "rows": 0,
        "bad_id": 0,
        "dup_prefix": 0,
    }
    seen_prefix = set()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows"] += 1
            id_raw = (row.get('ID') or '').strip()
            pref = _extract_prefix(id_raw)
            if not pref:
                stats["bad_id"] += 1
                continue
            if pref in seen_prefix:
                stats["dup_prefix"] += 1
                continue
            flip_prefixes.add(pref)
            seen_prefix.add(pref)

    return flip_prefixes, stats


def build_train_lists_from_csv(
    base_csv_path: str,
    variant_csv_path: str,
    flip_csv_path: Optional[str] = None,
    keep_order_by_base: bool = False,
    verbose: bool = True
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    return:
      - train_base: List[str]
      - train_variant: List[str]
      - report: 统计信息
    """
    base_map, base_stats = _load_id_stmt_map(base_csv_path)
    var_map,  var_stats  = _load_id_stmt_map(variant_csv_path)

    base_keys = set(base_map.keys())
    var_keys  = set(var_map.keys())
    
    common_before_flip = base_keys & var_keys

    if flip_csv_path:
        flip_prefixes, flip_stats = _load_flip_prefix_set(flip_csv_path)
    else:
        flip_prefixes, flip_stats = set(), {"rows": 0, "bad_id": 0, "dup_prefix": 0}

    common = common_before_flip - flip_prefixes

    if keep_order_by_base:
        ordered = []
        with open(base_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pref = _extract_prefix((row.get('ID') or '').strip())
                if pref and pref in common and pref not in ordered:
                    ordered.append(pref)
        prefix_list = ordered
    else:
        prefix_list = sorted(common)

    train_base    = [base_map[p][1] for p in prefix_list]
    train_variant = [var_map[p][1]  for p in prefix_list]

    blocked_by_flip = len(common_before_flip & flip_prefixes)
    report = {
        "base_rows": base_stats["rows"],
        "variant_rows": var_stats["rows"],
        "paired": len(prefix_list),
        "only_in_base_after_filter": len(base_keys - var_keys),
        "only_in_variant_after_filter": len(var_keys - base_keys),
        "bad_id_base": base_stats["bad_id"],
        "bad_id_variant": var_stats["bad_id"],
        "empty_stmt_base": base_stats["empty_stmt"],
        "empty_stmt_variant": var_stats["empty_stmt"],
        "dup_prefix_base": base_stats["dup_prefix"],
        "dup_prefix_variant": var_stats["dup_prefix"],
        "flip_rows": flip_stats["rows"],
        "bad_id_flip": flip_stats["bad_id"],
        "dup_prefix_flip": flip_stats["dup_prefix"],
        "blocked_by_flip": blocked_by_flip,
    }

    if verbose:
        print(f"[CSV] base rows={report['base_rows']} (bad_id={report['bad_id_base']}, empty_stmt={report['empty_stmt_base']}, dup={report['dup_prefix_base']})")
        print(f"[CSV] variant rows={report['variant_rows']} (bad_id={report['bad_id_variant']}, empty_stmt={report['empty_stmt_variant']}, dup={report['dup_prefix_variant']})")
        if flip_csv_path:
            print(f"[CSV] flip rows={report['flip_rows']} (bad_id={report['bad_id_flip']}, dup={report['dup_prefix_flip']}); blocked_by_flip={report['blocked_by_flip']}")
        print(f"[CSV] paired(after flip filter)={report['paired']}, only_in_base_after_filter={report['only_in_base_after_filter']}, only_in_variant_after_filter={report['only_in_variant_after_filter']}")
        # 打印前 3 对样例（便于人工核对）
        for p in prefix_list[:3]:
            print(f"[CSV] sample pair prefix={p} | base_id={base_map[p][0]} | variant_id={var_map[p][0]}")

    return train_base, train_variant, report


class LinearProbe2Way(nn.Module):
    """Two-way linear probe with bias; logits shape [N, 2]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.W(feats)   # [N, 2]

@torch.no_grad()
def extract_feature_mean_hidden_at_layer(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    layer_idx: int = -1,   # which hidden layer to pool; -1=last
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]}
    ]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    dev = get_input_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}

    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_hidden_states=True,
        return_dict=True
    )
    hs = out.hidden_states[layer_idx]   # [1, seq, hidden]
    feat = hs.mean(dim=1).squeeze(0)   # [hidden]
    return feat.to(torch.float32).cpu()

# ==== Feature matrix cache ====
@torch.no_grad()
def extract_feature_matrix_cached(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str,
    texts: List[str],
    layer_idx: int,
    cache_fp: str | None = None,
) -> torch.Tensor:
    if cache_fp and os.path.exists(cache_fp):
        arr = np.load(cache_fp)  # float32 ndarray
        return torch.from_numpy(arr)

    feats = []
    for t in texts:
        v = extract_feature_mean_hidden_at_layer(
            model, processor, system_prompt, build_user_prompt(t), layer_idx=layer_idx
        )  # -> float32 CPU [hidden]
        feats.append(v)
    X = torch.stack(feats, dim=0).contiguous()  # [N, H] CPU
    if cache_fp:
        np.save(cache_fp, X.numpy())
    return X

@torch.no_grad()
def extract_feature_mean_hidden_at_layer_raw(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    text: str,
    layer_idx: int = -1,
    max_tokens: int = 1024,
) -> torch.Tensor:
    """
    用“原始文本（不带 chat 模板）”抽取第 layer_idx 层的 mean-pooled 特征。
    论文做法：对该层 hidden_states 的所有 token 在序列维求均值。
    """
    tok = processor.tokenizer
    dev = get_input_device(model)

    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    hs = out.hidden_states[layer_idx]   # [1, seq, hidden]
    feat = hs.mean(dim=1).squeeze(0)    # [hidden]
    return feat.to(torch.float32).cpu()



# ==== CE + epochs two-way probe with caching ====
class CEProbe(nn.Module):
    """2-way linear probe with bias; logits shape [N, 2]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # [N, 2]

def fit_or_load_probe_ce(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    texts_pos: List[str],
    texts_neg: List[str],
    system_prompt: str,
    probe_layers: Union[int, Tuple[int,int], List[int]] = -1,
    save_dir: str = "probe_ckpts",
    tag: str = "vote_ce",
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    seed: int = 0,
    input_mode: str = "chat",         # "chat" | "raw"
    raw_max_len: int = 1024,
    fuse_mode: str = "sum",
    fuse_weights: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    训练/加载 2-way CE 探针（**仅使用传入的 texts_pos/texts_neg** 作为训练集）。
    返回: delta_w, W2, b2, mu, sigma  (均在 CPU float32)
    同时保存 ckpt: {"W2","b2","mu","sigma","layer_idx","tag"}
    """
    os.makedirs(save_dir, exist_ok=True)
    lay_key = _layer_key(probe_layers if isinstance(probe_layers, (list, tuple)) else [probe_layers])
    fuse_key = f"{fuse_mode}" + ("" if fuse_weights is None else f"_{'-'.join(map(str, fuse_weights))}")
    probe_path = os.path.join(save_dir, f"probe_{tag}_{lay_key}_{input_mode}_{fuse_key}.pt")

    # desired_class_idx = 1

    # 若已有训练好的探针，直接加载（注意：此时默认这些权重/统计就是 train 切分学到的）
    if os.path.exists(probe_path):
        ckpt = torch.load(probe_path, map_location="cpu")
        W2 = ckpt["W2"].to(torch.float32)
        b2 = ckpt["b2"].to(torch.float32)
        
        mu_v  = ckpt.get("mu", None)
        sigma_v = ckpt.get("sigma", None)
        std_used = bool(ckpt.get("std_used", False))

        H = W2.shape[1]
        if mu_v is None:
            mu = torch.zeros(1, H, dtype=torch.float32)
        else:
            mu = mu_v.to(torch.float32).unsqueeze(0) if mu_v.ndim == 1 else mu_v.to(torch.float32)
        if sigma_v is None:
            sigma = torch.ones(1, H, dtype=torch.float32)
        else:
            sigma = sigma_v.to(torch.float32).unsqueeze(0) if sigma_v.ndim == 1 else sigma_v.to(torch.float32)

        W_target = W2[1]  # 目标类（pos/clean）的权重行
        if std_used:
            # 旧版：探针在标准化空间训练过 → 反标准化到原空间
            # sigma_vec = sigma.squeeze(0) + 1e-12
            delta_w = (W_target / (W_target.norm(p=2) + 1e-12)).cpu()
        else:
            # 新版（本函数训练法）：直接就是原空间权重
            delta_w = W_target.cpu()
        # delta_w = W2[1] / (W2[1].norm() + 1e-12)
        return delta_w.cpu(), W2.cpu(), b2.cpu(), mu.cpu(), sigma.cpu()

    # 1) 抽取 **train** 特征（按 split_name="train" 单独缓存）
    model.eval()
    X_pos = extract_feature_matrix_for_texts_cached(
        model, processor, system_prompt, texts_pos, probe_layers,
        save_dir, tag, split_name="train_pos",
        input_mode=input_mode, raw_max_len=raw_max_len,
        fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_neg = extract_feature_matrix_for_texts_cached(
        model, processor, system_prompt, texts_neg, probe_layers,
        save_dir, tag, split_name="train_neg",
        input_mode=input_mode, raw_max_len=raw_max_len,
        fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X = torch.cat([X_pos, X_neg], dim=0)  # [N,H]
    y = torch.cat([
        torch.ones(len(X_pos), dtype=torch.long),
        torch.zeros(len(X_neg), dtype=torch.long)
    ], dim=0)

    # 2) 标准化统计 **仅来自 train**
    # mu = X.mean(0, keepdim=True)
    # sigma = X.std(0, keepdim=True).clamp_min(1e-6)
    # X_std = (X - mu) / sigma

    H = X.shape[1]
    # 不做标准化：把 mu/sigma 设成 0/1（保证 evaluate_* 的“标准化”变成空操作）
    mu    = torch.zeros(1, H, dtype=torch.float32)
    sigma = torch.ones(1, H, dtype=torch.float32)

    # 3) CE + epoch 训练（可复现）
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    probe = CEProbe(H).cpu()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = probe(xb)     # [B,2]
            loss   = ce(logits, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        W2 = probe.fc.weight.detach().to(torch.float32)  # [2,H]
        b2 = probe.fc.bias.detach().to(torch.float32)    # [2]
        # delta_w = (W2[1] - W2[0]).cpu()
        # delta_w = W2[1] / (W2[1].norm() + 1e-12)
        # sigma_vec = sigma.squeeze(0) + 1e-12                          # [H]
        # delta_w = (W2[1] - W2[0]) / sigma_vec
        delta_w = W2[1]
        # delta_w = delta_w / (delta_w.norm(p=2) + 1e-12)

    # 4) 保存（便于复现）
    torch.save({
        "W2": W2.cpu(), "b2": b2.cpu(),
        "mu": mu.squeeze(0).cpu(), "sigma": sigma.squeeze(0).cpu(),
        "layers": list(probe_layers if isinstance(probe_layers, (list,tuple)) else [probe_layers]), "tag": tag,
        "std_used": False, "input_mode": input_mode, "raw_max_len": raw_max_len,
        "fuse_mode": fuse_mode, "fuse_weights": None if fuse_weights is None else list(map(float, fuse_weights)),
    }, probe_path)

    return delta_w.cpu(), W2.cpu(), b2.cpu(), mu.cpu(), sigma.cpu()

class ProbeEvalResult(NamedTuple):
    acc_train: float
    acc_val: float
    acc_test: float
    auroc_train: Optional[float]
    auroc_val: Optional[float]
    auroc_test: Optional[float]

def _softmax_logits_to_pred_and_prob(logits: torch.Tensor):
    # logits: [N, 2]
    probs = torch.softmax(logits, dim=-1)         # [N,2]
    pred  = torch.argmax(probs, dim=-1)           # [N]
    pos_p = probs[:, 1]                           # positive-class prob
    return pred, pos_p

def evaluate_probe_on_splits(
    W2: torch.Tensor, b2: torch.Tensor,
    mu: torch.Tensor, sigma: torch.Tensor,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor,   y_val: torch.Tensor,
    X_test: torch.Tensor,  y_test: torch.Tensor,
    compute_auroc: bool = False,
    save_json: Optional[str] = None,
) -> ProbeEvalResult:
    """
    使用保存下来的 (W2,b2,mu,sigma) 在三个切分上做前向并计算 accuracy（可选 AUROC）。
    所有输入均为 CPU float32 / long tensor。
    """
    # 标准化
    Xtr = (X_train - mu) / sigma
    Xva = (X_val   - mu) / sigma
    Xte = (X_test  - mu) / sigma

    # 前向（线性头）
    with torch.no_grad():
        logits_tr = Xtr @ W2.T + b2          # [Ntr,2]
        logits_va = Xva @ W2.T + b2          # [Nva,2]
        logits_te = Xte @ W2.T + b2          # [Nte,2]

        pred_tr, pos_tr = _softmax_logits_to_pred_and_prob(logits_tr)
        pred_va, pos_va = _softmax_logits_to_pred_and_prob(logits_va)
        pred_te, pos_te = _softmax_logits_to_pred_and_prob(logits_te)

    acc_tr = float((pred_tr == y_train).float().mean().item())
    acc_va = float((pred_va == y_val).float().mean().item())
    acc_te = float((pred_te == y_test).float().mean().item())

    if compute_auroc:
        try:
            au_tr = float(roc_auc_score(y_train.numpy(), pos_tr.numpy()))
            au_va = float(roc_auc_score(y_val.numpy(),   pos_va.numpy()))
            au_te = float(roc_auc_score(y_test.numpy(),  pos_te.numpy()))
        except Exception:
            au_tr = au_va = au_te = None
    else:
        au_tr = au_va = au_te = None

    res = ProbeEvalResult(acc_tr, acc_va, acc_te, au_tr, au_va, au_te)

    if save_json is not None:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump({
                "acc_train": res.acc_train, "acc_val": res.acc_val, "acc_test": res.acc_test,
                "auroc_train": res.auroc_train, "auroc_val": res.auroc_val, "auroc_test": res.auroc_test,
            }, f, ensure_ascii=False, indent=2)

    return res

def split_texts_balanced(
    texts_pos: List[str],
    texts_neg: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    先对正类、负类各自独立随机打乱并按比例切分，再合并（保证类平衡、可复现）。
    返回: pos_tr, pos_va, pos_te, neg_tr, neg_va, neg_te（全是文本列表）
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    def _split_one(lst: List[str]):
        idx = torch.randperm(len(lst), generator=g).tolist()
        lst = [lst[i] for i in idx]
        n_tr = int(len(lst) * train_ratio)
        n_va = int(len(lst) * val_ratio)
        return lst[:n_tr], lst[n_tr:n_tr+n_va], lst[n_tr+n_va:]

    pos_tr, pos_va, pos_te = _split_one(texts_pos)
    neg_tr, neg_va, neg_te = _split_one(texts_neg)
    return pos_tr, pos_va, pos_te, neg_tr, neg_va, neg_te

@torch.no_grad()
def extract_feature_matrix_for_texts_cached(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str,
    texts: List[str],
    probe_layers: Union[int, Tuple[int,int], List[int]],  # 单层或多层
    save_dir: str,
    tag: str,
    split_name: str,                 # "train_pos" / "train_neg" / ...
    input_mode: str = "chat",        # "chat" 或 "raw"
    raw_max_len: int = 1024,
    fuse_mode: str = "sum",          # "sum" | "mean" | "fixed"
    fuse_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    返回: [N, H] (CPU float32)。若 probe_layers 是单个 int，等价于你原来的实现。
    若是多个层，则先取每层 mean-pooled hidden，再做 H 维融合（sum/mean/fixed）。
    """
    os.makedirs(save_dir, exist_ok=True)
    layers = probe_layers if isinstance(probe_layers, (list, tuple)) else [int(probe_layers)]
    lay_key = _layer_key(layers)
    fuse_key = f"{fuse_mode}" + ("" if fuse_weights is None else f"_{'-'.join(map(str, fuse_weights))}")
    cache_fp = os.path.join(save_dir, f"feats_{tag}_{split_name}_{lay_key}_{input_mode}_{fuse_key}.npy")
    if os.path.exists(cache_fp):
        arr = np.load(cache_fp)
        return torch.from_numpy(arr)

    feats = []
    for t in texts:
        ml_feats = _extract_hidden_for_text(
            model, processor, t, layers=layers, input_mode=input_mode,
            system_prompt=system_prompt, raw_max_len=raw_max_len
        )  # list of [H]
        f = _fuse_hdim(ml_feats, mode=fuse_mode, weights=fuse_weights)  # [H]
        feats.append(f)
    X = torch.stack(feats, dim=0).contiguous()  # [N,H]
    np.save(cache_fp, X.numpy())
    return X

def load_or_make_feature_caches_for_probe(
    model, processor, system_prompt,
    texts_pos: List[str], texts_neg: List[str],
    layer_idx: int, save_dir: str, tag: str
):
    os.makedirs(save_dir, exist_ok=True)
    feats_pos_cache = os.path.join(save_dir, f"feats_pos_{tag}_L{layer_idx}.npy")
    feats_neg_cache = os.path.join(save_dir, f"feats_neg_{tag}_L{layer_idx}.npy")

    X_pos = extract_feature_matrix_cached(model, processor, system_prompt, texts_pos, layer_idx, feats_pos_cache)
    X_neg = extract_feature_matrix_cached(model, processor, system_prompt, texts_neg, layer_idx, feats_neg_cache)
    return X_pos, X_neg, feats_pos_cache, feats_neg_cache


@dataclass
class GateRowRef:
    layer_idx: int
    row_idx: int
    cos_sim: float
    module: Gemma3MLP  # 直接指向该层 MLP 模块（含 gate_proj）

def get_gemma3_mlp_layers(model: Gemma3ForConditionalGeneration) -> List[Tuple[int, Gemma3MLP]]:
    layers = []
    for i, name, block in get_decoder_layers(model):
        for subname, sub in block.named_modules():
            if isinstance(sub, Gemma3MLP):
                layers.append((i, sub))
                break
    if not layers:
        raise RuntimeError("No Gemma3MLP found; check transformers version.")
    return layers

def select_inactive_gate_rows(
    model: Gemma3ForConditionalGeneration,
    behavior_vec: torch.Tensor,        # 用 Δw 作为行为方向
    k_total: Optional[int] = None,
    k_per_layer: Optional[int] = 128,
) -> List[GateRowRef]:
    """
    在每层 gate_proj.weight 的行向量中，找和 Δw 余弦最小（最反向, ~-1）的行：
    这就是论文说的“通常在不良状态下 inactive 的向量”，对它们做 +alpha*Δw 激活最有效。
    """
    v_cpu = behavior_vec.detach().to('cpu', dtype=torch.float32)
    mlps = get_gemma3_mlp_layers(model)
    per_layer_candidates: List[GateRowRef] = []

    def row_cos(W1: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        W1f = W1.to(dtype=torch.float32)    # [out, in]
        num = torch.mv(W1f, v)              # [out]
        den = W1f.norm(dim=1) * (v.norm() + 1e-12)
        return num / (den + 1e-12)

    for lidx, mlp in mlps:
        if not hasattr(mlp, "gate_proj") or not isinstance(mlp.gate_proj, nn.Linear):
            continue
        row_param = mlp.gate_proj.weight
        dev   = row_param.device
        v_dev = v_cpu.to(device=dev, non_blocking=True)
        cos   = row_cos(row_param.data, v_dev)  # on dev
        if k_per_layer is not None:
            vals, idxs = torch.topk(cos, k=k_per_layer, largest=False)  # 最小余弦（越负越好）
            vals = vals.detach().cpu().tolist()
            idxs = idxs.detach().cpu().tolist()
            for val, ridx in zip(vals, idxs):
                per_layer_candidates.append(GateRowRef(layer_idx=lidx, row_idx=ridx, cos_sim=float(val), module=mlp))
        else:
            cos_cpu = cos.detach().cpu().tolist()
            for ridx, val in enumerate(cos_cpu):
                per_layer_candidates.append(GateRowRef(layer_idx=lidx, row_idx=ridx, cos_sim=float(val), module=mlp))

    if k_total is not None:
        per_layer_candidates.sort(key=lambda r: r.cos_sim)  # 升序：更负在前
        return per_layer_candidates[:k_total]
    return per_layer_candidates

@contextlib.contextmanager
def model_surgery_context(
    selections: List[GateRowRef],
    behavior_vec: torch.Tensor,   # Δw
    alpha: float = 1.0,
):
    """
    临时参数编辑：对选中的 gate_proj 行执行  w_i <- w_i + alpha * Δw
    多 GPU 安全：每个设备缓存一份 Δw。
    """
    if len(selections) == 0:
        yield
        return
    backups = []
    w_cache: Dict[torch.device, torch.Tensor] = {}
    try:
        for ref in selections:
            row_param = ref.module.gate_proj.weight  # [out, in]
            dev, dtype = row_param.device, row_param.dtype
            if dev not in w_cache:
                w_cache[dev] = behavior_vec.to(device=dev, dtype=dtype, non_blocking=True)
            backups.append((row_param, ref.row_idx, row_param.data[ref.row_idx].detach().clone()))
            row_param.data[ref.row_idx].add_(alpha * w_cache[dev])  # 加法编辑
        yield
    finally:
        for row_param, ridx, buf in backups:
            row_param.data[ridx].copy_(buf)

def _get_model_max_len(processor: AutoProcessor, fallback: int = 2048) -> int:
    try:
        L = int(getattr(processor.tokenizer, "model_max_length", fallback))
        if L is None or L <= 0 or L > 100_000:  # 某些tokenizer返回超大占位
            return fallback
        return min(L, 8192)  # Gemma 3-4B 通常 8k；保守起见限制到 8k
    except Exception:
        return fallback

@torch.no_grad()
def evaluate_wikitext_ppl(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    dataset_config: str = "wikitext-2-raw-v1",     # 可改成 "wikitext-103-v1"
    split: str = "test",
    block_size: Optional[int] = None,              # None -> 用 tokenizer 最大长度或 2048
    stride: Optional[int] = None,                  # None -> 与 block_size 相同（不重叠）
    max_texts: Optional[int] = 200,                # 为了跑得快，默认只取前 200 段；设 None 用全量
) -> Tuple[float, float, int]:
    """
    返回: (avg_nll, ppl, counted_tokens)
    注：直接用 tokenizer（不走 chat 模板），标准 Causal LM loss（模型内部 shift）。
    """
    tok = processor.tokenizer
    dev = get_input_device(model)
    ds = load_dataset("wikitext", dataset_config, split=split)
    if max_texts is not None:
        ds = ds.select(range(min(max_texts, len(ds))))

    if block_size is None:
        block_size = _get_model_max_len(processor, 2048)
    if stride is None:
        stride = block_size

    total_neglog = 0.0
    total_tok = 0

    for ex in ds:
        text = ex["text"]
        if not text or not text.strip():
            continue
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(dev)
        if ids.numel() < 2:
            continue
        # 以 block_size 切块；stride=block_size 表示不重叠
        for i in range(0, max(0, ids.size(0) - 1), stride):
            chunk = ids[i:i + block_size]
            if chunk.numel() < 2:
                break
            inp = chunk.unsqueeze(0)
            out = model(input_ids=inp, labels=inp)
            loss = out.loss  # 已对有效 token 求平均
            n_tok = chunk.numel()
            total_neglog += float(loss.item()) * n_tok
            total_tok += int(n_tok)

    if total_tok == 0:
        return float("nan"), float("nan"), 0
    avg_nll = total_neglog / total_tok
    ppl = float(math.exp(avg_nll))
    return avg_nll, ppl, total_tok

def _choice_token_ids(tokenizer, choices=("A","B","C","D")) -> List[int]:
    ids = []
    for c in choices:
        x = tokenizer.encode(c, add_special_tokens=False)
        if len(x) != 1:
            raise ValueError(f"Choice '{c}' not single-token for this tokenizer: {x}")
        ids.append(x[0])
    return ids

def _format_mmlu_user_prompt(question: str, choices: List[str]) -> str:
    # MMLU标准多选：只输出字母
    lines = [question.strip()]
    labels = ["A", "B", "C", "D"]
    for lab, ch in zip(labels, choices):
        lines.append(f"{lab}. {ch}")
    lines.append("Answer with the letter of the correct option. Only output A, B, C, or D.")
    lines.append("Answer:")
    return "\n".join(lines)

@torch.no_grad()
def evaluate_mmlu_zero_shot(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str,
    subjects: Optional[List[str]] = None,   # None -> 全部科目
    split: str = "validation",              # 常见取法；也可用 "test"（部分环境无答案）
    max_examples_per_subject: Optional[int] = 50,  # 为了速度，默认每科最多50题
) -> Tuple[float, int, int]:
    """
    返回: (overall_acc, correct, total)
    做法：对每题构造 Chat 输入，取下一token在 {A,B,C,D} 上的概率，选最大者，与GT对比。
    """
    dev = get_input_device(model)
    tok = processor.tokenizer
    choice_ids = _choice_token_ids(tok, ("A","B","C","D"))

    if subjects is None:
        # MMLU 的新加载方式 - 使用 "cais/mmlu" 或者列出所有子集
        try:
            # 尝试方法1：直接加载所有配置
            from datasets import get_dataset_config_names
            subjects = get_dataset_config_names("cais/mmlu")
            if "all" in subjects:
                subjects.remove("all")  # 移除 "all" 配置
        except Exception:
            # 方法2：手动指定一些常见科目
            subjects = [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                "clinical_knowledge", "college_biology", "college_chemistry",
                "college_computer_science", "college_mathematics", "college_medicine",
                "college_physics", "computer_security", "conceptual_physics",
                "econometrics", "electrical_engineering", "elementary_mathematics",
                "formal_logic", "global_facts", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science",
                "high_school_european_history", "high_school_geography",
                "high_school_government_and_politics", "high_school_macroeconomics",
                "high_school_mathematics", "high_school_microeconomics",
                "high_school_physics", "high_school_psychology",
                "high_school_statistics", "high_school_us_history",
                "high_school_world_history", "human_aging", "human_sexuality",
                "international_law", "jurisprudence", "logical_fallacies",
                "machine_learning", "management", "marketing", "medical_genetics",
                "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
                "philosophy", "prehistory", "professional_accounting",
                "professional_law", "professional_medicine", "professional_psychology",
                "public_relations", "security_studies", "sociology", "us_foreign_policy",
                "virology", "world_religions"
            ]
    
    correct = 0
    total = 0

    for subj in subjects:
        try:
            # 修改：使用新的数据集路径
            ds = load_dataset("cais/mmlu", subj, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load subject {subj}: {e}")
            continue
            
        if max_examples_per_subject is not None:
            ds = ds.select(range(min(max_examples_per_subject, len(ds))))
        
        for ex in ds:
            # 字段名可能是: question, choices, answer
            q = ex.get("question", ex.get("input"))
            ch = ex.get("choices", ex.get("options"))
            ans = ex.get("answer", ex.get("target"))
            
            if q is None or ch is None or ans is None:
                continue
                
            # 处理答案格式
            if isinstance(ans, str):
                ans = ans.strip()
                gt_idx = {"A":0, "B":1, "C":2, "D":3}.get(ans.upper(), None)
            else:
                gt_idx = int(ans)
                
            if gt_idx is None or gt_idx < 0 or gt_idx > 3:
                continue

            user_text = _format_mmlu_user_prompt(q, ch)
            messages = [
                {"role": "system", "content": [{"type":"text", "text": system_prompt}]},
                {"role": "user",   "content": [{"type":"text", "text": user_text}]}
            ]
            enc = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_tensors="pt", return_dict=True
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)
            
            answer_pos = enc["input_ids"].shape[-1] - 1
            logits = out.logits[:, answer_pos, :].squeeze(0)
            cand_logits = logits.index_select(dim=-1, index=torch.tensor(choice_ids, device=logits.device))
            probs = torch.softmax(cand_logits, dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            correct += int(pred_idx == gt_idx)
            total += 1

    acc = float(correct / total) if total > 0 else float("nan")
    return acc, correct, total


def run_model_surgery_once(
    base_texts: List[str],         # “正类”（比如 non-toxic 或 agree）的句子集合
    variant_texts: List[str],      # “负类”（比如 toxic 或 disagree）的句子集合
    eval_pair: Tuple[str, str],    # (clean, corrupt)
    probe_layers: Union[int, Tuple[int,int], List[int]] = (27, 32),     # 你也可以试 -2 / 31 等（论文在多个层试过）
    alpha_grid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0),
    k_per_layer: int = 128,        # 每层选多少个行向量（Gemma-3-4B默认128比较稳）
    also_sweep_per_layer_alpha: bool = True,
    use_raw_text_for_probe: bool = False,   # << 新增：训练探头是否用原始文本
    raw_max_len_for_probe: int = 1024,
    fuse_mode: str = "sum",                   # <<< 新增：两层融合方式（H 维）
    fuse_weights: Optional[Sequence[float]] = None,
):
    assert len(base_texts) > 0 and len(variant_texts) > 0
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    ).eval()

    # ---- (A) 文本级切分（先切分，再训练探针）----
    pos_tr, pos_va, pos_te, neg_tr, neg_va, neg_te = split_texts_balanced(
        base_texts, variant_texts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0
    )
    # num_pairs = min(len(base_texts), len(variant_texts))
    # idx = torch.randperm(num_pairs, generator=torch.Generator().manual_seed(0)).tolist()
    # n_tr = int(num_pairs * 0.8)
    # n_va = int(num_pairs * 0.1)
    # idx_tr = idx[:n_tr]
    # idx_va = idx[n_tr:n_tr+n_va]
    # idx_te = idx[n_tr+n_va:]

    # pos_tr = [base_texts[i] for i in idx_tr]
    # pos_va = [base_texts[i] for i in idx_va]
    # pos_te = [base_texts[i] for i in idx_te]
    # neg_tr = [variant_texts[i] for i in idx_tr]
    # neg_va = [variant_texts[i] for i in idx_va]
    # neg_te = [variant_texts[i] for i in idx_te]

    # 1) 训练两类行为探针（CE），得到 Δw
    delta_w, W2, b2, mu, sigma = fit_or_load_probe_ce(
        model, processor,
        texts_pos=pos_tr, texts_neg=neg_tr,
        system_prompt=SYSTEM_PROMPT,
        probe_layers=probe_layers,
        save_dir="probe_ckpts",     # 可自定义
        tag="stance_invariance",    # 区分不同数据/任务
        epochs=25, lr=1e-4, batch_size=16, weight_decay=0.0, seed=0,
        input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe,
        fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )

    # ---- (C) 抽取 train/val/test 的特征（各自 split 名缓存），并评估探针 ACC/AUROC ----
    X_pos_tr = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, pos_tr, probe_layers, "probe_ckpts", "stance_invariance", "train_pos", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_neg_tr = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, neg_tr, probe_layers, "probe_ckpts", "stance_invariance", "train_neg", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_pos_va = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, pos_va, probe_layers, "probe_ckpts", "stance_invariance", "val_pos", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_neg_va = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, neg_va, probe_layers, "probe_ckpts", "stance_invariance", "val_neg", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_pos_te = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, pos_te, probe_layers, "probe_ckpts", "stance_invariance", "test_pos", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )
    X_neg_te = extract_feature_matrix_for_texts_cached(
        model, processor, SYSTEM_PROMPT, neg_te, probe_layers, "probe_ckpts", "stance_invariance", "test_neg", input_mode=("raw" if use_raw_text_for_probe else "chat"),
        raw_max_len=raw_max_len_for_probe, fuse_mode=fuse_mode, fuse_weights=fuse_weights
    )

    lay_key = _layer_key(probe_layers if isinstance(probe_layers, (list, tuple)) else [probe_layers])
    # fuse_key = f"{fuse_mode}" + ("" if fuse_weights is None else f"_{'-'.join(map(str, fuse_weights))}")

    def _pack(Xp, Xn):
        X = torch.cat([Xp, Xn], dim=0).float()
        y = torch.tensor([1]*len(Xp) + [0]*len(Xn), dtype=torch.long)
        # 打乱一下（可复现）
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        perm = torch.randperm(len(y), generator=g)
        return X[perm], y[perm]

    Xtr, ytr = _pack(X_pos_tr, X_neg_tr)
    Xva, yva = _pack(X_pos_va, X_neg_va)
    Xte, yte = _pack(X_pos_te, X_neg_te)

    eval_res = evaluate_probe_on_splits(
        W2.float(), b2.float(), mu.float(), sigma.float(),
        Xtr, ytr, Xva, yva, Xte, yte,
        compute_auroc=(roc_auc_score is not None),
        save_json=os.path.join("probe_ckpts", f"probe_eval_stance_invariance_L{lay_key}.json")
    )
    print(f"[Probe ACC] train={eval_res.acc_train:.3f}  val={eval_res.acc_val:.3f}  test={eval_res.acc_test:.3f}")
    if eval_res.auroc_test is not None:
        print(f"[Probe AUROC] train={eval_res.auroc_train:.3f}  val={eval_res.auroc_val:.3f}  test={eval_res.auroc_test:.3f}")

    # 2) 选择“通常在不良状态下不激活”的行向量（和 Δw 余弦最小）
    selections = select_inactive_gate_rows(model, delta_w, k_total=17408, k_per_layer=k_per_layer)
    # 只在高层做手术：最后 8 层
    # all_layers = get_decoder_layers(model)
    # n_layers = len(all_layers)
    # ALLOWED_LAYERS = set(range(n_layers - 22, n_layers))  # 也可改成 -6 或 -10

    # selections = [s for s in selections if s.layer_idx in ALLOWED_LAYERS]
    # print(f"[Length of selections] {len(selections)}")

    # === General capability baseline ===
    base_nll, base_ppl, base_tok = evaluate_wikitext_ppl(
        model, processor,
        dataset_config="wikitext-2-raw-v1",  # 你要全量可改 "wikitext-103-v1"
        split="test", block_size=None, stride=None, max_texts=200  # 为了速度先抽样
    )
    print(f"[WikiText] baseline: NLL={base_nll:.4f}, PPL={base_ppl:.3f}, toks={base_tok}")

    # mmlu_acc_before, mmlu_ok, mmlu_total = evaluate_mmlu_zero_shot(
    #     model, processor, MMLU_SYSTEM_PROMPT,
    #     subjects=None, split="validation", max_examples_per_subject=50
    # )
    # print(f"[MMLU] baseline zero-shot acc={mmlu_acc_before:.3f} ({mmlu_ok}/{mmlu_total})")

    # 3) 评估（沿用你现有的 objective）
    enc_clean   = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(eval_pair[0]))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(eval_pair[1]))
    with torch.no_grad():
        logits_clean   = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)
    obj_clean   = objective_from_logits_full(logits_clean,   enc_clean,   clean_probs, TEMP_FOR_PROBS).item()
    obj_corrupt = objective_from_logits_full(logits_corrupt, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()

    # 3a) 全层合并编辑下的最佳 alpha
    best = None
    best_probs = None
    best_ppl = None
    for a in alpha_grid:
        with model_surgery_context(selections, delta_w, alpha=a):
            logits_patched = forward_logits_only(model, enc_corrupt)
            patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

            # 通用侧：WikiText PPL
            patched_nll, patched_ppl, _ = evaluate_wikitext_ppl(
                model, processor,
                dataset_config="wikitext-2-raw-v1", split="test",
                block_size=None, stride=None, max_texts=200
            )
        obj_patched = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
        denom = obj_clean - obj_corrupt
        r_js = normalized_restoration(js_divergence, clean_probs, corrupt_probs, patched_probs).item()
        r_w = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs).item()
        r = float("nan") if abs(denom) < 1e-9 else (obj_patched - obj_corrupt) / denom

        ppl_increase_pct = (patched_ppl / base_ppl - 1.0) * 100.0 if base_ppl == base_ppl else float("inf")
        # 如果你需要硬约束（比如不得上升 >5%），加一行：
        pass_constraint = (ppl_increase_pct <= 20.0)
        # pass_constraint = True

        # if (best is None) or (not math.isnan(r_w) and r_w > best[0]):
        #     best = (r_w, a)
        #     best_probs = patched_probs
        if pass_constraint and (best is None or (not math.isnan(r_w) and r_w > best[0])):
            best = (r_w, a)
            best_probs = patched_probs
            best_ppl = patched_ppl

        print(f"[Alpha {a:.2f}] R_W1={r_w:.3f} | WikiText PPL={patched_ppl:.3f} (Δ={ppl_increase_pct:.1f}%) NLL={patched_nll:.3f}")
        
    if best is None:
        print("No valid alpha found.")
        return
    r_all, a_all = best
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt probs]   {corrupt_probs}")
    print(f"[Patched logits] {best_probs}")
    print(f"[ModelSurgery] best alpha (all-layers) = {a_all:.3f}, restoration={r_all:.3f}")

    if best_ppl is not None:
        delta_pct = (best_ppl / base_ppl - 1.0) * 100.0
        print(f"[WikiText] baseline PPL={base_ppl:.3f} → patched PPL={best_ppl:.3f} (Δ={delta_pct:.1f}%)")
    
    # with model_surgery_context(selections, delta_w, alpha=a_all):
    #     mmlu_acc_after, ok2, tot2 = evaluate_mmlu_zero_shot(
    #         model, processor, MMLU_SYSTEM_PROMPT,
    #         subjects=None, split="validation", max_examples_per_subject=50
    #     )
    # print(f"[MMLU] after-surgery zero-shot acc={mmlu_acc_after:.3f} ({ok2}/{tot2}); Δ={(mmlu_acc_after - mmlu_acc_before)*100:.1f} pts")

    # 3b) 分层报告（用相同 a_all）——可与 3c) 对比
    # per_layer_same_alpha = []
    # sel_map: Dict[int, List[GateRowRef]] = {}
    # for ref in selections:
    #     sel_map.setdefault(ref.layer_idx, []).append(ref)

    # for l in sorted(sel_map.keys()):
    #     with model_surgery_context(sel_map[l], delta_w, alpha=a_all):
    #         logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    #     obj_patched = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
    #     denom = obj_clean - obj_corrupt
    #     # r_l = normalized_restoration(js_divergence, clean_probs, corrupt_probs, patched_probs).item()
    #     r_w = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs).item()
    #     per_layer_same_alpha.append((l, r_w))
    # per_layer_same_alpha.sort(key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
    # print("[Per-layer @ same alpha] (top 20)")
    # for i, (l, r_w) in enumerate(per_layer_same_alpha[:20], 1):
    #     txt = "nan" if math.isnan(r_w) else f"{r_w:.3f}"
    #     print(f" #{i:02d} layer={l:02d} restoration={txt}")

    # 3c)（可选）逐层各自扫 alpha，解释你之前看到的“分层更高”的现象
    # if also_sweep_per_layer_alpha:
    #     per_layer_best = []
    #     for l in sorted(sel_map.keys()):
    #         best_l = None
    #         for a in alpha_grid:
    #             with model_surgery_context(sel_map[l], delta_w, alpha=a):
    #                 logits_patched = forward_logits_only(model, enc_corrupt)
    #             patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    #             obj_patched = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
    #             denom = obj_clean - obj_corrupt
    #             # r_l = normalized_restoration(js_divergence, clean_probs, corrupt_probs, patched_probs).item()
    #             r_w = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs).item()
    #             if (best_l is None) or (not math.isnan(r_w) and r_w > best_l[0]):
    #                 best_l = (r_w, a)
    #         if best_l is not None:
    #             per_layer_best.append((l, best_l[0], best_l[1]))
    #     per_layer_best.sort(key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
    #     print("[Per-layer best alpha] (top 20)")
    #     for i, (l, r_w, a_l) in enumerate(per_layer_best[:20], 1):
    #         txt = "nan" if math.isnan(r_w) else f"{r_w:.3f}"
    #         print(f" #{i:02d} layer={l:02d} best_alpha={a_l:.3f} restoration={txt}")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    torch.set_grad_enabled(True)

    set_global_determinism(0, single_thread=True)

    # print("=== Baseline diagnostics: activation patching / ablation ===")
    # _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)

    print("\n=== Paper-faithful model surgery (activate typically inactive vectors) ===")
    # For real experiments, expand these lists to dozens/hundreds of pairs.
    BASE_CSV_PATH    = "data/original_statements.csv"
    VARIANT_CSV_PATH = "data/it-clefts_variants.csv"
    FLIP_CSV_PATH = "data/flip rate/it-clefts_flip_4B.csv"

    train_base, train_variant, rep = build_train_lists_from_csv(
        BASE_CSV_PATH, VARIANT_CSV_PATH, FLIP_CSV_PATH,
        keep_order_by_base=False,  # 或 True：按 base CSV 顺序
        verbose=True
    )
    if len(train_base) == 0:
        raise RuntimeError("从 CSV 没配出任何成对样本：可能是 ID 格式不匹配、两边无共同前缀，或 statement 为空。")

    eval_pair = (BASE_TEXT, VARIANT_TEXT)
    run_model_surgery_once(train_base, train_variant, eval_pair, k_per_layer=10240)
