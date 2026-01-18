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

# ---------------------------
# Config
# ---------------------------

model_name = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# CUDA_VISIBLE_DEVICES=1,2 python Intervention/parameter_edit.py

# Example pair for base vs variant (you can change these)
# BASE_TEXT = "The government should abolish the ban on face-covering clothing."
# VARIANT_TEXT = "It is the ban on face-covering clothing that the government should abolish."

# BASE_TEXT = "Houses should be built on land currently used for agriculture."
# VARIANT_TEXT = "It is on land currently used for agriculture that Houses should be built."

BASE_TEXT = "Primary school teachers should earn as much as secondary school teachers."
VARIANT_TEXT = "It is as much as secondary school teachers that primary school teachers should earn."

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

# No. 10
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

# BASE_TEXT = "Parties should strive for a closer ratio of men to women when drawing up lists."
# VARIANT_TEXT = "It is a closer ratio of men to women that Parties should strive for when drawing up lists."

# No. 20
# BASE_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation."
# VARIANT_TEXT = "It is inflation that a price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight."

# BASE_TEXT = "Italy should get out of the Eurozone."
# VARIANT_TEXT = "It is the Eurozone that Italy should get out of."

# BASE_TEXT = "European economic integration has gone too far: member states should regain more autonomy."
# VARIANT_TEXT = "It is European economic integration that has gone too far: it is member states that should regain more autonomy."

# BASE_TEXT = "Restrictions on personal freedom and privacy are acceptable to deal with health emergencies such as Covid-19."
# VARIANT_TEXT = "It is health emergencies such as Covid-19 that restrictions on personal freedom and privacy are acceptable to deal with."

# BASE_TEXT = "Drilling is necessary to find more energy resources."
# VARIANT_TEXT = "It is more energy resources that drilling is necessary to find."

# BASE_TEXT = "The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."
# VARIANT_TEXT = "It is the hospital offering (national hospital planning with regard to locations and range of services) that the federal government should be given the authority to determine."

# BASE_TEXT = "There should be tax cuts at the federal level over the next four years."
# VARIANT_TEXT = "It is tax cuts that there should be at the federal level over the next four years."

# BASE_TEXT = "There should be stricter controls on equal pay for women and men."
# VARIANT_TEXT = "It is stricter controls on equal pay for women and men that there should be."

# BASE_TEXT = "To achieve climate targets, incentives and target agreements should be relied on exclusively, rather than bans and restrictions."
# VARIANT_TEXT = "It is incentives and target agreements that should be relied on exclusively, rather than bans and restrictions, to achieve climate targets."

# BASE_TEXT = "The army's target number of soldiers should expand to at least 120,000."
# VARIANT_TEXT = "It is at least 120,000 that the army's target number of soldiers should expand to."

# No. 30
# BASE_TEXT = "The Swiss Armed Forces should expand their cooperation with NATO."
# VARIANT_TEXT = "It is their cooperation with NATO that the Swiss Armed Forces should expand."

# BASE_TEXT = "Switzerland should terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons."
# VARIANT_TEXT = "It is the Bilateral Agreements with the EU that Switzerland should terminate and it is a free trade agreement without the free movement of persons that Switzerland should seek."

# BASE_TEXT = "Switzerland should return to a strict interpretation of neutrality (renounce economic sanctions to a large extent)."
# VARIANT_TEXT = "It is a strict interpretation of neutrality that Switzerland should return to (renounce economic sanctions to a large extent)."

topk_attr = 6          # how many top layers to print/consider in diagnostics
print_top_layers = 34  # how many top layers to print
TEMP_FOR_PROBS = 1.0
EPS = 1e-9

# ---------------------------
# Utilities / Model Introspection
# ---------------------------

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
def attn_ablation_23(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layer_to_edit: int = 23,
    ratio: float = 0.0
):
    with attn_ablation_context(
        model,
        enc,
        layers_to_edit=[layer_to_edit],
        ratio=ratio,
    ):
        yield

@contextlib.contextmanager
def attn_ablation_23_16(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ratio: float = 0.0
):
    with attn_ablation_context(
        model,
        enc,
        layers_to_edit=[16, 23],
        ratio=ratio,
    ):
        yield

@contextlib.contextmanager
def attn_head_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    heads_to_edit: List[int],
    ratio: float = 0.0,
    all_positions: bool = False,
):
    """
    在给定的层里，把若干 attention head 的输出缩放为 ratio。
    - ratio=0.0 就是“把这个 head 设为 0”。
    - all_positions=True: 所有 token 位置都 ablate
      all_positions=False: 只在 enc.answer_pos 这个位置 ablate（更接近你现在的做法）
    """
    hooks = []

    # 这里假设 Gemma3 的 config 里有 hidden_size 和 num_attention_heads
    num_heads = model.config.text_config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            # module 是 Gemma3Attention 里的 o_proj
            if layer_idx not in layers_to_edit:
                return output

            # inputs[0] 是 o_proj 的输入：形状 [batch, seq, hidden_size]，
            # 实际上就是 concat 后的所有 head
            x = inputs[0]
            B, T, H = x.shape
            head_dim = H // num_heads

            # [B, T, H] -> [B, T, num_heads, head_dim]
            x = x.view(B, T, num_heads, head_dim)

            # 做一个 clone 避免就地修改带来奇怪的梯度 / 共享引用问题
            # x = x.clone()

            if all_positions:
                for h_idx in heads_to_edit:
                    if 0 <= h_idx < num_heads:
                        x[:, :, h_idx, :] = x[:, :, h_idx, :] * ratio
            else:
                pos = enc.answer_pos
                for h_idx in heads_to_edit:
                    if 0 <= h_idx < num_heads:
                        x[:, pos, h_idx, :] = x[:, pos, h_idx, :] * ratio

            # 再 reshape 回去
            x = x.view(B, T, H)

            # 手动走一次线性层，相当于 o_proj(x)
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
def attn_head_ablation_scaling_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    heads_to_ablate: List[int],
    heads_to_scaling: List[int],
    ablate_ratio: float = 0.0,
    scaling_ratio: float = 2.0,
    all_positions: bool = False,
):
    """
    在给定的层里，把若干 attention head 的输出缩放为 ratio。
    - ratio=0.0 就是“把这个 head 设为 0”。
    - all_positions=True: 所有 token 位置都 ablate
      all_positions=False: 只在 enc.answer_pos 这个位置 ablate（更接近你现在的做法）
    """
    hooks = []

    # 这里假设 Gemma3 的 config 里有 hidden_size 和 num_attention_heads
    num_heads = model.config.text_config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            # module 是 Gemma3Attention 里的 o_proj
            if layer_idx not in layers_to_edit:
                return output

            # inputs[0] 是 o_proj 的输入：形状 [batch, seq, hidden_size]，
            # 实际上就是 concat 后的所有 head
            x = inputs[0]
            B, T, H = x.shape
            head_dim = H // num_heads

            # [B, T, H] -> [B, T, num_heads, head_dim]
            x = x.view(B, T, num_heads, head_dim)

            # 做一个 clone 避免就地修改带来奇怪的梯度 / 共享引用问题
            # x = x.clone()

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

            # 再 reshape 回去
            x = x.view(B, T, H)

            # 手动走一次线性层，相当于 o_proj(x)
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
def attn_head_ablation_23(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ratio: float = 0.0,
    all_positions: bool = False,
    head: int = 0
):
    """
    只在第 23 层，把某个 head 的输出乘上 ratio。
    默认 all_positions=False，也就是只在 answer_pos ablate。
    """
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
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    heads: List[int],
    ratio: float = 0.0,
    all_positions: bool = False
):
    """
    只在第 23 层，把某个 head 的输出乘上 ratio。
    默认 all_positions=False，也就是只在 answer_pos ablate。
    """
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
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ablate_ratio: float = 0.0,
    scaling_ratio: float = 2.0,
    all_positions: bool = False
):
    with attn_head_ablation_scaling_context(
        model,
        enc,
        layers_to_edit=[23],
        heads_to_ablate=[1, 3, 6, 7],
        heads_to_scaling = [0, 2, 4],
        ablate_ratio=ablate_ratio,
        scaling_ratio=scaling_ratio,
        all_positions=all_positions
    ):
        yield

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

# def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-3):
#     d0 = dist_fn(p_clean, p_corrupt)
#     dp = dist_fn(p_clean, p_patched)
#     # R = (dp - d0) / (dp + d0 + eps)
#     R = (dp-d0) / 3.0
#     # return torch.where(d0 <= eps, torch.full_like(R, float('nan')), R)
#     return R

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
    
    # def sweep_attn_patch(kind: str):
    #     results = []
    #     best_r = 0.6
    #     best_patched_probs = None
    #     ppl_increase_pct = 0.0
    #     for l in range(n_layers):
    #         spec = {"block": [], "attn": [], "mlp": []}
    #         spec[kind] = [l]

    #         with patch_context(model, enc_corrupt, clean_cache, spec):
    #             logits_patched = forward_logits_only(model, enc_corrupt)
    #             patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    #             r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    #             if r > best_r:
    #                 best_patched_probs = patched_probs
    #                 best_r = r
    #                 patched_nll, patched_ppl, _ = evaluate_wikitext_ppl(
    #                     model, processor,
    #                     dataset_config="wikitext-2-raw-v1", split="test",
    #                     block_size=None, stride=None, max_texts=200
    #                 )
    #                 ppl_increase_pct = (patched_ppl / base_ppl - 1.0) * 100.0
            
    #         obj_patched = objective_from_logits_full(
    #             logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
    #         ).item()
    #         # denom = obj_clean - obj_corrupt
    #         # if abs(denom) < 1e-9:
    #         #     r = float("nan")
    #         # else:
    #         #     r = (obj_patched - obj_corrupt) / denom
    #         results.append((l, r))
    #     return results, best_patched_probs, ppl_increase_pct


    block_results= sweep_patch("block")
    attn_results= sweep_patch("attn")
    mlp_results= sweep_patch("mlp")

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


    def sweep_layer_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        best_ppl = None
        for l in range(n_layers):
            with block_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            obj_patched = objective_from_logits_full(
                logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
            ).item()
            results.append((l, r))
        return results, best_patched_probs, best_ppl

    layer_ablate_results, layer_best_probs, best_ppl= sweep_layer_ablate(ratio=0.0)
    print(f"[Ablate-BLOCK Best-Patched-Probs] {layer_best_probs}")
    # print(f"[Ablate-ATTN Best Delta PPL] {best_ppl}")
    print("-" * 60)
    print_top("[Ablate-BLOCK ratio=0.0] top layers", layer_ablate_results)

    with block_ablation_context(model, enc_corrupt, layers_to_edit=[15], ratio=0.0, pos_strategy="last"):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
        r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    
    print(f"[Ablate Block-15] {patched_probs}")
    print(f"Restoration Score: {r}")
    print("-" * 60)


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
                    # patched_nll, patched_ppl, _ = evaluate_wikitext_ppl(
                    #     model, processor,
                    #     dataset_config="wikitext-2-raw-v1", split="test",
                    #     block_size=None, stride=None, max_texts=200
                    # )
                    # ppl_increase_pct = (patched_ppl / base_ppl - 1.0) * 100.0
                    # best_ppl = ppl_increase_pct


            obj_patched = objective_from_logits_full(
                logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
            ).item()
            # denom = obj_clean - obj_corrupt
            # if abs(denom) < 1e-9:
            #     r = float("nan")
            # else:
            #     r = (obj_patched - obj_corrupt) / denom
            # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            # if r > best_r:
            #     best_r = r
            #     best_patched_probs = patched_probs
            #     best_ppl = ppl_increase_pct
            results.append((l, r))
        return results, best_patched_probs, best_ppl

    ablate0_results, best_probs, best_ppl= sweep_attn_ablate(ratio=0.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs}")
    # print(f"[Ablate-ATTN Best Delta PPL] {best_ppl}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=0.0] top layers", ablate0_results)

    ablate_results, best_probs_1, best_ppl_1= sweep_attn_ablate(ratio=2.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs_1}")
    # print(f"[Ablate-ATTN Best Delta PPL] {best_ppl}")
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
                    # patched_nll, patched_ppl, _ = evaluate_wikitext_ppl(
                    #     model, processor,
                    #     dataset_config="wikitext-2-raw-v1", split="test",
                    #     block_size=None, stride=None, max_texts=200
                    # )
                    # ppl_increase_pct = (patched_ppl / base_ppl - 1.0) * 100.0
                    # best_ppl = ppl_increase_pct


            obj_patched = objective_from_logits_full(
                logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS
            ).item()
            # denom = obj_clean - obj_corrupt
            # if abs(denom) < 1e-9:
            #     r = float("nan")
            # else:
            #     r = (obj_patched - obj_corrupt) / denom
            # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            # if r > best_r:
            #     best_r = r
            #     best_patched_probs = patched_probs
            #     best_ppl = ppl_increase_pct
            results.append((l, r))
        return results, best_patched_probs, best_ppl

    mlp_ablate0_results, mlp_best_probs, best_ppl= sweep_mlp_ablate(ratio=0.0)
    print(f"[Ablate-MLP Best-Patched-Probs] {mlp_best_probs}")
    # print(f"[Ablate-ATTN Best Delta PPL] {best_ppl}")
    print("-" * 60)
    print_top("[Ablate-MLP ratio=0.0] top layers", mlp_ablate0_results)

    # with attn_ablation_23(model, enc_corrupt, layer_to_edit=23, ratio=0.0):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_ablation_23_16(model, enc_corrupt, ratio=0.0):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23-and-16 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # 对注意力头进行patch
    def sweep_head_ablate(
        ratio: float = 0.0,
        all_positions: bool = False,
    ):
        results = []
        best_r = 0.0
        best_patched_probs = None
        num_heads = model.config.text_config.num_attention_heads

        for h in range(num_heads):
            with attn_head_ablation_23(
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
    
    print_top_head(f"[Ablate-ATTN-23-HEADS]", head_results)

    with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[1, 3, 6, 7], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Ablate ATTN-23 Head-1-3-6-7 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

    with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[1, 3, 6, 7], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Scaling up ATTN-23 Head-1-3-6-7 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

    with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[0, 2, 4, 5], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Ablate ATTN-23 Head-0-2-4-5 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

    with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[0, 2, 4], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Scaling-up ATTN-23 Head-0-2-4 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

    with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=-3.0, heads=[0, 2, 4], all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[Only Neg-Scaling ATTN-23 Head-0-2-4 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

    with attn_head_ablation_scaling_multiple_23(model, enc_corrupt, scaling_ratio=2.5, all_positions=False):
        logits_patched = forward_logits_only(model, enc_corrupt)
        patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    print(f"[ATTN-23 Ablate-Head-1-3-6-7 Scaling-Head-0-2-4 R-Score] {r}")
    print(f"[Patched Probs] {patched_probs}")

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


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    torch.set_grad_enabled(True)

    set_global_determinism(0, single_thread=True)

    print("=== Baseline diagnostics: activation patching / ablation ===")
    _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)
