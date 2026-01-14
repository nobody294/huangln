import os
import math
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3Attention,
    Gemma3MLP
)

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# Put all flip pairs of ONE wording rule here (for aggregation)
PAIRS: List[Tuple[str, str]] = [
    ("The government should abolish the ban on face-covering clothing.",
     "The ban on face-covering clothing should be abolished by the government."),

    ("Primary school teachers should earn as much as secondary school teachers.",
     "As much as secondary school teachers earn should be earned by primary school teachers."),

    ("The Netherlands should introduce an additional flight tax for short-distance flights.",
     "An additional flight tax for short-distance flights should be introduced by the Netherlands."),

    ("An increase in minimum wages should no longer automatically result in an increase in welfare benefits.",
     "An increase in welfare benefits should no longer automatically be the result of an increase in minimum wages."),

    ("All employed persons are to be required to be insured in the statutory pension scheme.",
     "The government is to require all employed persons to be insured in the statutory pension scheme."),

    ("Donations from companies to political parties should continue to be permitted.",
     "Relative authorities should continue to permit donations from companies to political parties."),

    ("Chinese companies should not be allowed to receive contracts for the expansion of the communications infrastructure in Germany.",
     "Contracts for the expansion of the communications infrastructure in Germany should not be allowed to be received by Chinese companies."),

    ("A tax is to be levied again on high assets.",
     "The government is to levy a tax on high assets again."),

    ("Married couples without children should continue to receive tax breaks.",
     "Tax breaks should continue to be received by married couples without children."),

    ("Air traffic is to be taxed more heavily.",
     "The government is to tax air traffic more heavily."),

    ("The state should build low-rent apartments for rent.",
     "Low-rent apartments should be built by the state for rent."),

    ("Hungary should decide by referendum whether to remain part of the EU.",
     "Whether to remain part of the EU should be decided by Hungary by referendum."),

    ("Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities).",
     "Environmental influences (e.g. media content, sensitising activities) can influence gender identity."),

    ("Hungary should join the European Public Prosecutor's Office.",
     "The European Public Prosecutor's Office should be joined by Hungary."),

    ("Political influence has been reduced by changing the university model (reorganisation into a trust).",
     "Changing the university model (reorganisation into a trust) has reduced political influence."),

    ("Parties should strive for a closer ratio of men to women when drawing up lists.",
     "A closer ratio of men to women should be striven for by parties when lists are drawn up."),

    ("A law is needed to prevent companies from relocating their production abroad.",
     "Relative authorities need to pass a law to prevent companies from relocating their production abroad."),

    ("A heritage tax one's wealth should be introduced.",
     "Relative authorities should introduce a heritage tax one's wealth."),

    ("Compulsory vaccination of children should be introduced based on the Swiss vaccination plan.",
     "The government should introduce compulsory vaccination of children based on the Swiss vaccination plan."),

    ("Automatic facial recognition should be banned in public spaces.",
     "The government should ban automatic facial recognition in public spaces."),

    ("Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border.",
     "The Schengen agreement with the EU should be terminated by Switzerland and more security checks directly on the border should be reintroduced."),
]

RULE_NAME = "Active/Passive Conversions"

TEMP_FOR_PROBS = 1.0
EPS = 1e-12

# For Appendix Fig A (head-combo ablation)
HEAD_COMBO_LAYER = 23
HEADS_TO_ABLATE = [1, 3, 6, 7]
HEAD_ABLATE_RATIO = 0.0


# ---------------------------
# Determinism
# ---------------------------
def set_global_determinism(seed: int = 0, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
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
# Utilities / Model Introspection
# ---------------------------
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
        raise RuntimeError("No Gemma3DecoderLayer found. Check transformers version/model class.")
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
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    enc = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    dev = get_input_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    answer_pos = input_ids.shape[-1] - 1

    digit_ids = []
    tok = processor.tokenizer
    for d in range(1, 8):
        ids = tok.encode(str(d), add_special_tokens=False)
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
def forward_logits_only(model: Gemma3ForConditionalGeneration, enc: EncodedChat) -> torch.Tensor:
    out = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    return out.logits[:, enc.answer_pos, :].squeeze(0)


def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def digit_probs_from_logits_full(
    logits_full: torch.Tensor,
    enc: EncodedChat,
    temperature: float = 1.0
) -> torch.Tensor:
    digits = digit_logit_slice(logits_full, enc.digit_ids)
    return torch.softmax(digits / temperature, dim=-1)


# ---------------------------
# Distances + Restoration
# ---------------------------
def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12) -> torch.Tensor:
    d0 = dist_fn(p_clean, p_corrupt)
    dp = dist_fn(p_clean, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float("nan")), R)


# ---------------------------
# Clean cache for patching
# ---------------------------
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


def collect_clean_cache(model: Gemma3ForConditionalGeneration, enc_clean: EncodedChat) -> CleanCache:
    cache = CleanCache()
    hooks = []

    def layer_hook(layer_idx):
        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.block_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def attn_hook(layer_idx):
        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.attn_out[layer_idx] = vec.cpu()
            return out
        return _hook

    def mlp_hook(layer_idx):
        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            vec = hidden[:, enc_clean.answer_pos, :].detach().squeeze(0).to(hidden.dtype)
            cache.mlp_out[layer_idx] = vec.cpu()
            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_hook(i)))
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Gemma3MLP):
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
    model: Gemma3ForConditionalGeneration,
    enc_corrupt: EncodedChat,
    cache: CleanCache,
    patch_spec: Dict[str, List[int]],
):
    hooks = []
    cache.to_device_like(enc_corrupt.input_ids)

    def replace_at_answer(hidden: torch.Tensor, vec: torch.Tensor):
        new_hidden = hidden.clone()
        new_hidden[:, enc_corrupt.answer_pos, :] = vec.to(hidden.dtype).to(hidden.device)
        return new_hidden

    def layer_patch_hook(layer_idx):
        def _hook(module, inp, out):
            if layer_idx not in patch_spec.get("block", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.block_out[layer_idx].to(hidden.device)
            new_hidden = replace_at_answer(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def attn_patch_hook(layer_idx):
        def _hook(module, inp, out):
            if layer_idx not in patch_spec.get("attn", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.attn_out[layer_idx].to(hidden.device)
            new_hidden = replace_at_answer(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    def mlp_patch_hook(layer_idx):
        def _hook(module, inp, out):
            if layer_idx not in patch_spec.get("mlp", []):
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            vec = cache.mlp_out[layer_idx].to(hidden.device)
            new_hidden = replace_at_answer(hidden, vec)
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        hooks.append(layer.register_forward_hook(layer_patch_hook(i)))
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Gemma3MLP):
                hooks.append(sub.register_forward_hook(mlp_patch_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Ablation (inference-time masking)
# ---------------------------
@contextlib.contextmanager
def block_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
):
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()
            new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
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
):
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()
            new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention) and i in layers_to_edit:
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
):
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, out):
            if layer_idx not in layers_to_edit:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            new_hidden = hidden.clone()
            new_hidden[:, enc.answer_pos, :] = new_hidden[:, enc.answer_pos, :] * ratio
            return (new_hidden, *out[1:]) if isinstance(out, tuple) else new_hidden
        return _hook

    for i, name, layer in get_decoder_layers(model):
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3MLP) and i in layers_to_edit:
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Attention head COMBO ablation (only keep heads [1,3,6,7])
# ---------------------------
@contextlib.contextmanager
def attn_head_combo_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layer_to_edit: int,
    heads_to_edit: List[int],
    ratio: float = 0.0,
):
    hooks = []
    num_heads = model.config.text_config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            # Only intervene in the target layer
            if layer_idx != layer_to_edit:
                return output

            x = inputs[0]  # [B, T, H]
            B, T, H = x.shape
            head_dim = H // num_heads
            x = x.view(B, T, num_heads, head_dim)

            pos = enc.answer_pos
            for h_idx in heads_to_edit:
                if 0 <= h_idx < num_heads:
                    x[:, pos, h_idx, :] = x[:, pos, h_idx, :] * ratio

            x = x.view(B, T, H)

            # Recompute linear output
            W = module.weight
            b = module.bias
            out = torch.nn.functional.linear(x, W, b)
            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        if i != layer_to_edit:
            continue
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention) and hasattr(sub, "o_proj"):
                hooks.append(sub.o_proj.register_forward_hook(make_o_proj_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Compute profiles for one pair
# ---------------------------
def patching_profile_for_pair(model, processor, base_text, variant_text):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    clean_cache = collect_clean_cache(model, enc_clean)
    layers = get_decoder_layers(model)
    n_layers = len(layers)

    def sweep(kind: str) -> np.ndarray:
        arr = np.full((n_layers,), np.nan, dtype=np.float64)
        for l in range(n_layers):
            spec = {"block": [], "attn": [], "mlp": []}
            spec[kind] = [l]
            with patch_context(model, enc_corrupt, clean_cache, spec):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            arr[l] = float(r.item())
        return arr

    block_arr = sweep("block")
    attn_arr = sweep("attn")
    mlp_arr = sweep("mlp")

    return {
        "block": block_arr,
        "attn": attn_arr,
        "mlp": mlp_arr,
        "n_layers": n_layers,
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
    }


def ablation_profile_for_pair(model, processor, base_text, variant_text, ratio: float = 0.0):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    n_layers = len(get_decoder_layers(model))

    def sweep_block() -> np.ndarray:
        arr = np.full((n_layers,), np.nan, dtype=np.float64)
        for l in range(n_layers):
            with block_ablation_context(model, enc_corrupt, [l], ratio=ratio):
                logits_ab = forward_logits_only(model, enc_corrupt)
                probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
            arr[l] = float(r.item())
        return arr

    def sweep_attn() -> np.ndarray:
        arr = np.full((n_layers,), np.nan, dtype=np.float64)
        for l in range(n_layers):
            with attn_ablation_context(model, enc_corrupt, [l], ratio=ratio):
                logits_ab = forward_logits_only(model, enc_corrupt)
                probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
            arr[l] = float(r.item())
        return arr

    def sweep_mlp() -> np.ndarray:
        arr = np.full((n_layers,), np.nan, dtype=np.float64)
        for l in range(n_layers):
            with mlp_ablation_context(model, enc_corrupt, [l], ratio=ratio):
                logits_ab = forward_logits_only(model, enc_corrupt)
                probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
            arr[l] = float(r.item())
        return arr

    return {
        "block": sweep_block(),
        "attn": sweep_attn(),
        "mlp": sweep_mlp(),
        "n_layers": n_layers,
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
    }


def head_combo_effect_for_pair(model, processor, base_text, variant_text):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    # Apply head-combo ablation at the chosen layer
    with attn_head_combo_ablation_context(
        model=model,
        enc=enc_corrupt,
        layer_to_edit=HEAD_COMBO_LAYER,
        heads_to_edit=HEADS_TO_ABLATE,
        ratio=HEAD_ABLATE_RATIO,
    ):
        logits_ab = forward_logits_only(model, enc_corrupt)
        ab_probs = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, ab_probs)

    return {
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
        "ab_probs": ab_probs.detach().float().cpu().numpy(),
        "restoration": float(r.item()),
    }


def head_sweep_restoration_for_pair(
    model,
    processor,
    base_text: str,
    variant_text: str,
    layer: int = 23,
    ratio: float = 0.0
):
    """
    Sweep all heads at `layer`, ablate ONE head at a time (ratio=0.0 by default),
    return per-head restoration in head-index order (0..num_heads-1).

    Returns:
        head_ids: List[int] = [0,1,...,H-1]
        head_restorations: np.ndarray shape [H]
        combo_restoration: float restoration for HEADS_TO_ABLATE combo
    """
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    num_heads = model.config.text_config.num_attention_heads

    head_ids = list(range(num_heads))
    head_restorations = np.full((num_heads,), np.nan, dtype=np.float64)

    # one-head-at-a-time
    for h in range(num_heads):
        with attn_head_combo_ablation_context(
            model=model,
            enc=enc_corrupt,
            layer_to_edit=layer,
            heads_to_edit=[h],
            ratio=ratio,
        ):
            logits_ab = forward_logits_only(model, enc_corrupt)
            probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

        r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
        head_restorations[h] = float(r.item())

    # combo heads [1,3,6,7] (your requirement)
    with attn_head_combo_ablation_context(
        model=model,
        enc=enc_corrupt,
        layer_to_edit=layer,
        heads_to_edit=HEADS_TO_ABLATE,
        ratio=ratio,
    ):
        logits_ab = forward_logits_only(model, enc_corrupt)
        probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

    combo_r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
    combo_restoration = float(combo_r.item())

    return head_ids, head_restorations, combo_restoration


def head_sweep_restoration_mean_over_pairs(
    model,
    processor,
    pairs: List[Tuple[str, str]],
    layer: int = 23,
    ratio: float = 0.0,
):
    """
    Compute per-head restoration for each pair, then average over all pairs.
    Returns head_ids, mean_head_restorations, mean_combo_restoration.
    """
    all_head_rest = []
    all_combo_rest = []

    # compute per-pair
    for i, (b, v) in enumerate(pairs, 1):
        print(f"  Head sweep pair {i}/{len(pairs)}")
        head_ids, head_rest, combo_r = head_sweep_restoration_for_pair(
            model=model,
            processor=processor,
            base_text=b,
            variant_text=v,
            layer=layer,
            ratio=ratio,
        )
        all_head_rest.append(head_rest)
        all_combo_rest.append(combo_r)

    # stack: [N, H] -> mean over N
    mat = np.stack(all_head_rest, axis=0)
    mean_head_rest = np.nanmean(mat, axis=0)

    mean_combo = float(np.nanmean(np.array(all_combo_rest, dtype=np.float64)))

    return head_ids, mean_head_rest, mean_combo


# ---------------------------
# Aggregation
# ---------------------------
def bootstrap_ci_median(data_2d: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    N, L = data_2d.shape
    med = np.nanmedian(data_2d, axis=0)

    boot = np.empty((n_boot, L), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot[b] = np.nanmedian(data_2d[idx], axis=0)

    lo = np.nanpercentile(boot, 100 * (alpha / 2), axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return med, lo, hi


def aggregate_profiles(profile_dicts: List[Dict[str, np.ndarray]]):
    n_layers = profile_dicts[0]["n_layers"]
    out = {"n_layers": n_layers, "n_pairs": len(profile_dicts)}

    for comp in ["block", "attn", "mlp"]:
        mat = np.stack([p[comp] for p in profile_dicts], axis=0)  # [N, L]
        mean = np.nanmean(mat, axis=0)  # [L]
        out[comp] = {"mean": mean}

    return out


# ---------------------------
# Plotting (one subplot each)
# ---------------------------
def plot_layerwise_subplot(title: str, stats: Dict, ylabel: str):
    L = stats["n_layers"]
    x = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    for comp in ["block", "attn", "mlp"]:
        y = stats[comp]["mean"]
        ax.plot(x, y, label=comp)

    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig



def plot_head_bars_by_index(
    title: str,
    head_ids: List[int],
    head_restorations: np.ndarray,
    combo_restoration: float,
    layer: int,
):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.2))  # 宽一点，不然标签挤
    x = np.array(head_ids, dtype=int)

    ax.bar(x, head_restorations)

    ax.set_title(
        f"{title}\nLayer {layer} per-head ablation restoration (by head index)\n"
        f"Combo heads {HEADS_TO_ABLATE} restoration: {combo_restoration:.3f}"
    )
    ax.set_xlabel("Head index")
    ax.set_ylabel("Normalized restoration score")

    # 关键：标出所有 head 的刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in head_ids], rotation=90, fontsize=7)

    # 组合 ablation 的水平虚线（作为对照）
    ax.axhline(combo_restoration, linestyle="--")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig




def plot_digit_probs_clean_corrupt_ablate(title: str, clean_probs, corrupt_probs, ab_probs, restoration_value: float):
    digits = np.arange(1, 8)
    width = 0.26

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    ax.bar(digits - width, clean_probs, width=width, label="Clean (base)")
    ax.bar(digits, corrupt_probs, width=width, label="Corrupt (variant)")
    ax.bar(digits + width, ab_probs, width=width, label=f"Head-ablate L{HEAD_COMBO_LAYER} heads {HEADS_TO_ABLATE}")

    ax.set_title(f"{title}\nNormalized restoration (W1-based): {restoration_value:.3f}")
    ax.set_xlabel("Likert digit")
    ax.set_ylabel("Probability")
    ax.set_xticks(digits)
    ax.set_ylim(0.0, max(clean_probs.max(), corrupt_probs.max(), ab_probs.max()) * 1.15 + 1e-6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


# ---------------------------
# Main
# ---------------------------
def main():
    set_global_determinism(0, single_thread=True)
    torch.set_grad_enabled(False)

    print("Loading model + processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    # ---------------------------
    # Appendix Fig A (one subplot): Head-combo ablation at layer 23 (heads 1,3,6,7)
    # For simplicity (and interpretability), we plot ONE representative pair:
    # the first pair in PAIRS.
    # ---------------------------
    print(f"\n[Appendix Fig A] Per-head ablation bar chart (by head index) at layer {HEAD_COMBO_LAYER}")
    
    head_ids, head_restorations, combo_r = head_sweep_restoration_mean_over_pairs(
        model=model,
        processor=processor,
        pairs=PAIRS,                 # <-- 关键：用全部 pairs
        layer=HEAD_COMBO_LAYER,
        ratio=HEAD_ABLATE_RATIO,
    )


    figA = plot_head_bars_by_index(
        title=f"Attention head ablation\n{RULE_NAME}",
        head_ids=head_ids,
        head_restorations=head_restorations,
        combo_restoration=combo_r,
        layer=HEAD_COMBO_LAYER,
    )
    figA.savefig("appendix_figA_head_bars_by_index.png", dpi=200)
    print("Saved: appendix_figA_head_bars_by_index.png")

    # ---------------------------
    # Figure 1 (one subplot): Activation patching curves
    # ---------------------------
    print(f"\n[Figure 1] Activation patching profiles for rule: {RULE_NAME}")
    patch_profiles = []
    for i, (b, v) in enumerate(PAIRS, 1):
        print(f"  Patching pair {i}/{len(PAIRS)}")
        patch_profiles.append(patching_profile_for_pair(model, processor, b, v))
    patch_stats = aggregate_profiles(patch_profiles)

    fig1 = plot_layerwise_subplot(
        title=f"Activation patching\n{RULE_NAME} (flip pairs n={patch_stats['n_pairs']})",
        stats=patch_stats,
        ylabel="Normalized restoration score",
    )
    fig1.savefig("fig1_patching_one_rule.png", dpi=200)
    print("Saved: fig1_patching_one_rule.png")

    # ---------------------------
    # Figure 2 (one subplot): Ablation curves (ratio=0.0)
    # ---------------------------
    print(f"\n[Figure 2] Ablation (ratio=0.0) profiles for rule: {RULE_NAME}")
    ab_profiles = []
    for i, (b, v) in enumerate(PAIRS, 1):
        print(f"  Ablation pair {i}/{len(PAIRS)}")
        ab_profiles.append(ablation_profile_for_pair(model, processor, b, v, ratio=0.0))
    ab_stats = aggregate_profiles(ab_profiles)

    fig2 = plot_layerwise_subplot(
        title=f"Inference-time masking (ratio=0.0)\n{RULE_NAME} (flip pairs n={ab_stats['n_pairs']})",
        stats=ab_stats,
        ylabel="Normalized restoration score",
    )
    fig2.savefig("fig2_ablation_one_rule.png", dpi=200)
    print("Saved: fig2_ablation_one_rule.png")

    # Show all three figures
    plt.show()


if __name__ == "__main__":
    main()
