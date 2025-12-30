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

model_name = "google/gemma-3-12b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

MMLU_SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. "
    "Answer correctly and output exactly one capital letter: A, B, C, or D."
)

# CUDA_VISIBLE_DEVICES=1,2 python Intervention/negation_edit_12B.py

# Example pair for base vs variant (you can change these)
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

topk_attr = 6          # how many top layers to print/consider in diagnostics
print_top_layers = 20  # how many top layers to print
TEMP_FOR_PROBS = 1.0
EPS = 1e-9

# ---------------------------
# Utilities / Model Introspection
# ---------------------------

def flip_probs_1_to_7(p: torch.Tensor) -> torch.Tensor:
    """
    p: [..., 7]，最后一维是 1..7 的概率。
    返回左右翻转后的分布（以 4 为中心镜像）。
    """
    idx = torch.tensor([6, 5, 4, 3, 2, 1, 0], device=p.device)
    return p.index_select(dim=-1, index=idx)

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
    p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
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

    neg_scaling_results, best_probs_2, best_ppl_1= sweep_attn_ablate(ratio=-1.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs_2}")
    # print(f"[Ablate-ATTN Best Delta PPL] {best_ppl}")
    print("-" * 60)
    print_top("[Neg-Scaling-ATTN ratio=-1.0] top layers", neg_scaling_results)


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
    # def sweep_head_ablate(
    #     ratio: float = 0.0,
    #     all_positions: bool = False,
    # ):
    #     results = []
    #     best_r = 0.0
    #     best_patched_probs = None
    #     num_heads = model.config.text_config.num_attention_heads

    #     for h in range(num_heads):
    #         with attn_head_ablation_23(
    #             model,
    #             enc_corrupt,
    #             ratio=ratio,
    #             all_positions=all_positions,
    #             head=h
    #         ):
    #             logits_patched = forward_logits_only(model, enc_corrupt)
    #             patched_probs = digit_probs_from_logits_full(
    #                 logits_patched, enc_clean, TEMP_FOR_PROBS
    #             )

    #         r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)

    #         if r > best_r:
    #             best_r = r
    #             best_patched_probs = patched_probs

    #         results.append((h, r))

    #     return results, best_patched_probs
    
    # head_results, best_head_probs = sweep_head_ablate(ratio=0.0, all_positions=False)
    # print(f"[Abalte-ATTN-Head Best Probs] {best_head_probs}")

    # def print_top_head(title, arr):
    #     arr_sorted = sorted(arr, key=lambda x: (0 if math.isnan(x[1]) else x[1]), reverse=True)
    #     print(title)
    #     for i, (h, r) in enumerate(arr_sorted[:print_top_layers], 1):
    #         txt = "nan" if math.isnan(r) else f"{r:.3f}"
    #         print(f" #{i:02d} head={h:02d} restoration={txt}")
    #     print("-" * 60)
    
    # print_top_head(f"[Ablate-ATTN-23-HEADS]", head_results)

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[1, 3, 6, 7], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23 Head-1-3-6-7 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[1, 3, 6, 7], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Scaling up ATTN-23 Head-1-3-6-7 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=0.0, heads=[0, 2, 4], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Ablate ATTN-23 Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=5.0, heads=[0, 2, 4], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Scaling-up ATTN-23 Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_edit_23_multiple_heads(model, enc_corrupt, ratio=-3.0, heads=[0, 2, 4], all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[Only Neg-Scaling ATTN-23 Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

    # with attn_head_ablation_scaling_multiple_23(model, enc_corrupt, scaling_ratio=2.5, all_positions=False):
    #     logits_patched = forward_logits_only(model, enc_corrupt)
    #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
    
    # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
    # print(f"[ATTN-23 Ablate-Head-1-3-6-7 Scaling-Head-0-2-4 R-Score] {r}")
    # print(f"[Patched Probs] {patched_probs}")

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
