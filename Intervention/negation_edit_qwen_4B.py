import os
import math
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen3 internals for layer/attn/mlp hooks
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3Attention,
        Qwen3MLP,
    )
except Exception as e:
    raise ImportError(
        "Failed to import Qwen3 internals from transformers. "
        "Please ensure transformers>=4.51.0 and that Qwen3 is supported.\n"
        f"Original error: {e}"
    )

# ---------------------------
# Config
# ---------------------------

model_name = "Qwen/Qwen3-4B"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# CUDA_VISIBLE_DEVICES=1,2 python Intervention/negation_edit_qwen_4B.py

# Example pair for base vs variant (you can change these)
# BASE_TEXT = "Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists."
# VARIANT_TEXT = "There should not be a tax per kilometer driven for motorists instead of the tax on car ownership."

# BASE_TEXT = "Instead of the existing health insurance companies, there should be a national healthcare fund for everyone."
# VARIANT_TEXT = "There should not be a national health care fund for everyone instead of the existing health insurance companies."

# BASE_TEXT = "Houses should be built on land currently used for agriculture."
# VARIANT_TEXT = "No housing should be built on land now used for agriculture."

# BASE_TEXT = "People who consider their lives complete should be able to receive assistance with suicide."
# VARIANT_TEXT = "People who consider their lives complete should not be able to get help with suicide."

# BASE_TEXT = "Limiting rights and freedoms is necessary to combat organized crime."
# VARIANT_TEXT = "It is not necessary to limit rights and freedoms to combat organized crime."

# BASE_TEXT = "Climate change is solely attributable to human action."
# VARIANT_TEXT = "Climate change is not solely attributable to human action."

# BASE_TEXT = "The ability of landlords to increase housing rents is to be more strictly limited by law."
# VARIANT_TEXT = "The ability of landlords to increase rents should not be more strictly limited by law."

# BASE_TEXT = "All employed persons are to be required to be insured in the statutory pension scheme."
# VARIANT_TEXT = "People in employment should not necessarily have to be insured under the statutory pension scheme."

# BASE_TEXT = "The traditional family of father, mother and children is to be promoted more strongly than other living arrangements."
# VARIANT_TEXT = "The traditional family of father, mother and children should not be promoted more than other cohabiting couples."

# No. 10
# BASE_TEXT = "Germany is to leave the European Union."
# VARIANT_TEXT = "Germany should not leave the European Union."

# BASE_TEXT = "Facial recognition software should be allowed to be used for video surveillance in public places."
# VARIANT_TEXT = "No facial recognition software should be used for video surveillance in public places."

# BASE_TEXT = "The statutory minimum wage is to be increased to at least 12 euros by 2022 at the latest."
# VARIANT_TEXT = "The statutory minimum wage should not be increased to at least 12 euros by 2022 at the latest."

# BASE_TEXT = "The result of any nationwide referendum should be binding regardless of turnout."
# VARIANT_TEXT = "The results of certain nationwide referendums should be binding depending on turnout."

# BASE_TEXT = "The share of defense spending in Poland's GDP should be further increased."
# VARIANT_TEXT = "The share of defense spending in Poland's GDP should not be increased further."

# BASE_TEXT = "Education spending should be increased to at least the OECD average of 5.2 per cent (GDP)."
# VARIANT_TEXT = "Spending on education should not be increased to the OECD average of 5.2 per cent (of GDP)."

# BASE_TEXT = "Only men and women should be allowed to marry."
# VARIANT_TEXT = "Marriages should not be exclusively between men and women."

# BASE_TEXT = "European integration is all in all a positive process."
# VARIANT_TEXT = "European integration is not an all-positive process."

# BASE_TEXT = "Separation of careers between judges and prosecutors should be introduced."
# VARIANT_TEXT = "Separation of careers between judges and prosecutors should not be introduced."

# BASE_TEXT = "For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should be eliminated."
# VARIANT_TEXT = "For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should not be eliminated."

# No. 20
# BASE_TEXT = "The Federal Council's ability to restrict private and economic life in the event of a pandemic should be more limited."
# VARIANT_TEXT = "The Federal Council's ability to restrict private and economic life in the event of a pandemic should not be more limited."

# BASE_TEXT = "The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."
# VARIANT_TEXT = "The federal government should not be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."

# BASE_TEXT = "There should be the introduction of a national inheritance tax on all inheritances over one million Swiss francs."
# VARIANT_TEXT = "There should not be the introduction of a national inheritance tax on all inheritances over one million Swiss francs."

# BASE_TEXT = "A minimum wage of CHF 4,000 for all full-time employees should be introduced."
# VARIANT_TEXT = "A minimum wage of CHF 4,000 for all full-time employees should not be introduced."

BASE_TEXT = "Direct payments should only be granted to farmers with proof of ecological performance."
VARIANT_TEXT = "Direct payments should not only be granted to farmers with proof of ecological performance."


topk_attr = 6          # how many top layers to print/consider in diagnostics
print_top_layers = 36  # Qwen3-4B has 36 layers; print top N
TEMP_FOR_PROBS = 1.0
EPS = 1e-9

# For Qwen3-4B (base): disable thinking to keep "next token is digit" probing consistent
QWEN_ENABLE_THINKING = False


# ---------------------------
# Utilities / Model Introspection
# ---------------------------

def flip_probs_1_to_7(p: torch.Tensor) -> torch.Tensor:
    """
    p: [..., 7] last dim corresponds to digits 1..7
    Return the mirrored distribution around 4 (reverse order).
    """
    idx = torch.tensor([6, 5, 4, 3, 2, 1, 0], device=p.device)
    return p.index_select(dim=-1, index=idx)


def get_input_device(model) -> torch.device:
    # Robust for device_map="auto"
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model) -> List[Tuple[int, str, nn.Module]]:
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen3DecoderLayer):
            layers.append((len(layers), name, mod))
    if not layers:
        raise RuntimeError(
            "No Qwen3DecoderLayer found via named_modules(). "
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


def _apply_qwen_chat_template_or_fallback(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> str:
    """
    Prefer Qwen chat_template if present; otherwise fallback to simple concatenation.
    """
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return text
    # Fallback (base models sometimes have no template)
    return system_prompt.strip() + "\n\n" + user_prompt


def encode_for_next_token(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> EncodedChat:
    text = _apply_qwen_chat_template_or_fallback(
        tokenizer, system_prompt, user_prompt, enable_thinking=enable_thinking
    )
    enc = tokenizer(text, return_tensors="pt")
    dev = get_input_device(model)

    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)

    seq_len = input_ids.shape[-1]
    answer_pos = seq_len - 1

    # Important: for many BPE tokenizers, " 1" is a single token but "1" is not.
    digit_ids: List[int] = []
    for d in range(1, 8):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(" " + str(d), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {d} is not a single token for this tokenizer.")
        digit_ids.append(ids[0])

    return EncodedChat(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_pos=answer_pos,
        digit_ids=digit_ids,
    )


@torch.no_grad()
def forward_logits_only(model, enc: EncodedChat) -> torch.Tensor:
    out = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = out.logits[:, enc.answer_pos, :].squeeze(0)
    return logits


def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def pick_target_digit_id(logits_clean_digits: torch.Tensor, digit_ids: List[int]) -> int:
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
    model,
    enc_clean: EncodedChat,
    enc_corrupt: EncodedChat,
    clean_probs: Optional[torch.Tensor],
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

    obj = objective_from_logits_full(logits_corrupt, enc_corrupt, clean_probs, TEMP_FOR_PROBS)

    model.zero_grad(set_to_none=True)
    obj.backward(retain_graph=False)

    for h in fwd_hooks + bwd_hooks:
        h.remove()

    scores = []
    device = logits_corrupt.device
    layer_ids = sorted(
        set(clean_cache.block_out.keys())
        & set(h_corrupt_pos.keys())
        & set(grad_pos.keys())
    )
    for l in layer_ids:
        hc = clean_cache.block_out[l].to(device)
        hr = h_corrupt_pos[l].to(device)
        g = grad_pos[l].to(device)
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


def collect_clean_cache(model, enc_clean: EncodedChat) -> CleanCache:
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
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(mlp_hook(i)))

    with torch.no_grad():
        _ = model(
            input_ids=enc_clean.input_ids,
            attention_mask=enc_clean.attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

    for h in hooks:
        h.remove()
    return cache


@contextlib.contextmanager
def patch_context(
    model,
    enc_corrupt: EncodedChat,
    cache: CleanCache,
    patch_spec: Dict[str, List[int]],
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
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(mlp_patch_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def block_ablation_context(
    model,
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
    model,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "last",
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
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def mlp_ablation_context(
    model,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "last",
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
            if isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_ablation_23(model, enc: EncodedChat, layer_to_edit: int = 23, ratio: float = 0.0):
    with attn_ablation_context(model, enc, layers_to_edit=[layer_to_edit], ratio=ratio):
        yield


@contextlib.contextmanager
def attn_ablation_23_16(model, enc: EncodedChat, ratio: float = 0.0):
    with attn_ablation_context(model, enc, layers_to_edit=[16, 23], ratio=ratio):
        yield


@contextlib.contextmanager
def attn_head_ablation_context(
    model,
    enc: EncodedChat,
    layers_to_edit: List[int],
    heads_to_edit: List[int],
    ratio: float = 0.0,
    all_positions: bool = False,
):
    """
    In given layers, scale selected attention heads' contribution (by scaling the o_proj input slice).
    ratio=0.0 => ablate that head.
    """
    hooks = []
    num_heads = model.config.num_attention_heads  # Qwen3 uses top-level config

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            if layer_idx not in layers_to_edit:
                return output

            x = inputs[0]  # [B, T, H]
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
            if isinstance(sub, Qwen3Attention) and hasattr(sub, "o_proj"):
                hooks.append(sub.o_proj.register_forward_hook(make_o_proj_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_head_ablation_scaling_context(
    model,
    enc: EncodedChat,
    layers_to_edit: List[int],
    heads_to_ablate: List[int],
    heads_to_scaling: List[int],
    ablate_ratio: float = 0.0,
    scaling_ratio: float = 2.0,
    all_positions: bool = False,
):
    hooks = []
    num_heads = model.config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            if layer_idx not in layers_to_edit:
                return output

            x = inputs[0]
            B, T, H = x.shape
            head_dim = H // num_heads
            x = x.view(B, T, num_heads, head_dim)

            if all_positions:
                for h_idx in heads_to_ablate:
                    if 0 <= h_idx < num_heads:
                        x[:, :, h_idx, :] = x[:, :, h_idx, :] * ablate_ratio
                for idx in heads_to_scaling:
                    if 0 <= idx < num_heads:
                        x[:, :, idx, :] = x[:, :, idx, :] * scaling_ratio
            else:
                pos = enc.answer_pos
                for h_idx in heads_to_ablate:
                    if 0 <= h_idx < num_heads:
                        x[:, pos, h_idx, :] = x[:, pos, h_idx, :] * ablate_ratio
                for idx in heads_to_scaling:
                    if 0 <= idx < num_heads:
                        x[:, pos, idx, :] = x[:, pos, idx, :] * scaling_ratio

            x = x.view(B, T, H)
            out = torch.nn.functional.linear(x, module.weight, module.bias)
            return out

        return _hook

    for i, name, layer in get_decoder_layers(model):
        if i not in layers_to_edit:
            continue
        for subname, sub in layer.named_modules():
            if isinstance(sub, Qwen3Attention) and hasattr(sub, "o_proj"):
                hooks.append(sub.o_proj.register_forward_hook(make_o_proj_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def attn_head_ablation_23(
    model,
    enc: EncodedChat,
    ratio: float = 0.0,
    all_positions: bool = False,
    head: int = 0,
):
    with attn_head_ablation_context(
        model,
        enc,
        layers_to_edit=[23],
        heads_to_edit=[head],
        ratio=ratio,
        all_positions=all_positions,
    ):
        yield


@contextlib.contextmanager
def attn_head_edit_23_multiple_heads(
    model,
    enc: EncodedChat,
    heads: List[int],
    ratio: float = 0.0,
    all_positions: bool = False,
):
    with attn_head_ablation_context(
        model,
        enc,
        layers_to_edit=[23],
        heads_to_edit=heads,
        ratio=ratio,
        all_positions=all_positions,
    ):
        yield


@contextlib.contextmanager
def attn_head_ablation_scaling_multiple_23(
    model,
    enc: EncodedChat,
    ablate_ratio: float = 0.0,
    scaling_ratio: float = 2.0,
    all_positions: bool = False,
):
    with attn_head_ablation_scaling_context(
        model,
        enc,
        layers_to_edit=[23],
        heads_to_ablate=[1, 3, 6, 7],
        heads_to_scaling=[0, 2, 4],
        ablate_ratio=ablate_ratio,
        scaling_ratio=scaling_ratio,
        all_positions=all_positions,
    ):
        yield


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps=1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12):
    p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float("nan")), R)


# ---------------------------
# Main experiment: activation patching / ablation
# ---------------------------

def run_activation_patching(base_text: str, variant_text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    enc_clean = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(base_text), enable_thinking=QWEN_ENABLE_THINKING
    )
    enc_corrupt = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(variant_text), enable_thinking=QWEN_ENABLE_THINKING
    )

    with torch.no_grad():
        logits_clean = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
        logits_clean_digits = digit_logit_slice(logits_clean, enc_clean.digit_ids)
        logits_corrupt_digits = digit_logit_slice(logits_corrupt, enc_corrupt.digit_ids)

    target_digit_id = pick_target_digit_id(logits_clean_digits, enc_clean.digit_ids)
    clean_target_logit = logits_clean[target_digit_id].item()
    corrupt_target_logit = logits_corrupt[target_digit_id].item()

    c_id = pick_target_digit_id(logits_corrupt_digits, enc_corrupt.digit_ids)

    print(f"[Target digit id] {target_digit_id}  ({tokenizer.decode([target_digit_id])})")
    print(f"[Clean logits] {logits_clean_digits}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[c_id] {c_id}  ({tokenizer.decode([c_id])})")
    print("-" * 60)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    obj_clean = objective_from_logits_full(logits_clean, enc_clean, clean_probs, TEMP_FOR_PROBS).item()
    obj_corrupt = objective_from_logits_full(logits_corrupt, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()

    print(f"[Target digit id] {target_digit_id}  ({tokenizer.decode([target_digit_id])})  (for reference)")
    print(f"[Clean target logit]   {clean_target_logit:.3f}  (ref)")
    print(f"[Corrupt target logit] {corrupt_target_logit:.3f}  (ref)")
    print(f"[Clean logits] {logits_clean_digits}")
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt logits] {logits_corrupt_digits}")
    print(f"[Corrupt probs] {corrupt_probs}")
    print("-" * 60)

    # Optional: first-order attribution (needs backward)
    # scores_sorted = attribution_scores_first_order(model, enc_clean, enc_corrupt, clean_probs)
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
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)

            _ = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, float(r)))
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
    print_top("[Patch - MLP  - top layers]", mlp_results)

    def sweep_layer_ablate(ratio: float = 0.0):
        results = []
        best_r = -1e9
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with block_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                r_val = float(r)
                if r_val > best_r:
                    best_r = r_val
                    best_patched_probs = patched_probs
            results.append((l, r_val))
        return results, best_patched_probs, best_ppl

    layer_ablate_results, layer_best_probs, _ = sweep_layer_ablate(ratio=0.0)
    print(f"[Ablate-BLOCK Best-Patched-Probs] {layer_best_probs}")
    print("-" * 60)
    print_top("[Ablate-BLOCK ratio=0.0] top layers", layer_ablate_results)

    with block_ablation_context(model, enc_corrupt, layers_to_edit=[31], ratio=0.0, pos_strategy="last"):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
        r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
        r_value = float(r)
    
    print(f"[Ablate-BLOCK-31 Patched-Probs] {patched_probs}")
    print(f"Restoration Score: {r_value}")
    print("-" * 60)


    def sweep_attn_ablate(ratio: float = 0.0):
        results = []
        best_r = -1e9
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with attn_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                r_val = float(r)
                if r_val > best_r:
                    best_r = r_val
                    best_patched_probs = patched_probs
            results.append((l, r_val))
        return results, best_patched_probs, best_ppl

    ablate0_results, best_probs, _ = sweep_attn_ablate(ratio=0.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=0.0] top layers", ablate0_results)

    ablate_results, best_probs_1, _ = sweep_attn_ablate(ratio=2.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs_1}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=2.0] top layers", ablate_results)

    def sweep_mlp_ablate(ratio: float = 0.0):
        results = []
        best_r = -1e9
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with mlp_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                r_val = float(r)
                if r_val > best_r:
                    best_r = r_val
                    best_patched_probs = patched_probs
            results.append((l, r_val))
        return results, best_patched_probs, best_ppl

    mlp_ablate0_results, mlp_best_probs, _ = sweep_mlp_ablate(ratio=0.0)
    print(f"[Ablate-MLP Best-Patched-Probs] {mlp_best_probs}")
    print("-" * 60)
    print_top("[Ablate-MLP ratio=0.0] top layers", mlp_ablate0_results)

    # --------- Head-level ablation on layer 23 ---------

    # def sweep_head_ablate(ratio: float = 0.0, all_positions: bool = False):
    #     results = []
    #     best_r = -1e9
    #     best_patched_probs = None
    #     num_heads = model.config.num_attention_heads

    #     for h in range(num_heads):
    #         with attn_head_ablation_23(
    #             model,
    #             enc_corrupt,
    #             ratio=ratio,
    #             all_positions=all_positions,
    #             head=h,
    #         ):
    #             logits_patched = forward_logits_only(model, enc_corrupt)
    #             patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)

    #         r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    #         r_val = float(r)
    #         if r_val > best_r:
    #             best_r = r_val
    #             best_patched_probs = patched_probs

    #         results.append((h, r_val))

    #     return results, best_patched_probs

    # head_results, best_head_probs = sweep_head_ablate(ratio=0.0, all_positions=False)
    # print(f"[Ablate-ATTN-Head Best Probs] {best_head_probs}")

    # def print_top_head(title, arr):
    #     arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
    #     print(title)
    #     for i, (h, r) in enumerate(arr_sorted[:print_top_layers], 1):
    #         txt = "nan" if math.isnan(r) else f"{r:.3f}"
    #         print(f" #{i:02d} head={h:02d} restoration={txt}")
    #     print("-" * 60)

    # print_top_head("[Ablate-ATTN-23-HEADS]", head_results)

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[1, 3, 6, 7], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23 Head-1-3-6-7 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[1, 3, 6, 7], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Scaling up ATTN-23 Head-1-3-6-7 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[0, 2, 4, 5], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23 Head-0-2-4-5 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[0, 2, 4], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Scaling-up ATTN-23 Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=-3.0, heads=[0, 2, 4], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Neg-Scaling ATTN-23 Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_ablation_scaling_multiple_23(model, enc_corrupt, scaling_ratio=2.5, all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[ATTN-23 Ablate-Head-1-3-6-7 Scaling-Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    return True


# ---------------------------
# Paper-faithful pipeline helpers (determinism)
# ---------------------------

def set_global_determinism(seed: int = 42, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # or ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)


# ---------------------------
# Entry
# ---------------------------

if __name__ == "__main__":
    torch.set_grad_enabled(True)
    set_global_determinism(0, single_thread=True)

    print("=== Baseline diagnostics: activation patching / ablation (Qwen3-4B) ===")
    _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)
