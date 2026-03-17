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

model_name = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)


# Example pair for base vs variant
# BASE_TEXT = "The government should abolish the ban on face-covering clothing."
# VARIANT_TEXT = "The ban on face-covering clothing should be abolished by the government."

# BASE_TEXT = "Primary school teachers should earn as much as secondary school teachers."
# VARIANT_TEXT = "As much as secondary school teachers earn should be earned by primary school teachers."

# BASE_TEXT = "The Netherlands should introduce an additional flight tax for short-distance flights."
# VARIANT_TEXT = "An additional flight tax for short-distance flights should be introduced by the Netherlands."

# BASE_TEXT = "An increase in minimum wages should no longer automatically result in an increase in welfare benefits."
# VARIANT_TEXT = "An increase in welfare benefits should no longer automatically be the result of an increase in minimum wages."

# BASE_TEXT = "All employed persons are to be required to be insured in the statutory pension scheme."
# VARIANT_TEXT = "The government is to require all employed persons to be insured in the statutory pension scheme."

# BASE_TEXT = "Donations from companies to political parties should continue to be permitted."
# VARIANT_TEXT = "Relative authorities should continue to permit donations from companies to political parties."

# BASE_TEXT = "Chinese companies should not be allowed to receive contracts for the expansion of the communications infrastructure in Germany."
# VARIANT_TEXT = "Contracts for the expansion of the communications infrastructure in Germany should not be allowed to be received by Chinese companies."

# BASE_TEXT = "A tax is to be levied again on high assets."
# VARIANT_TEXT = "The government is to levy a tax on high assets again."

# BASE_TEXT = "Married couples without children should continue to receive tax breaks."
# VARIANT_TEXT = "Tax breaks should continue to be received by married couples without children."

# No. 10
# BASE_TEXT = "Air traffic is to be taxed more heavily."
# VARIANT_TEXT = "The government is to tax air traffic more heavily."

# BASE_TEXT = "The state should build low-rent apartments for rent."
# VARIANT_TEXT = "Low-rent apartments should be built by the state for rent."

# BASE_TEXT = "Hungary should decide by referendum whether to remain part of the EU."
# VARIANT_TEXT = "Whether to remain part of the EU should be decided by Hungary by referendum."

# BASE_TEXT = "Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities)."
# VARIANT_TEXT = "Environmental influences (e.g. media content, sensitising activities) can influence gender identity."

# BASE_TEXT = "Hungary should join the European Public Prosecutor's Office."
# VARIANT_TEXT = "The European Public Prosecutor's Office should be joined by Hungary."

# BASE_TEXT = "Political influence has been reduced by changing the university model (reorganisation into a trust)."
# VARIANT_TEXT = "Changing the university model (reorganisation into a trust) has reduced political influence."

# BASE_TEXT = "Parties should strive for a closer ratio of men to women when drawing up lists."
# VARIANT_TEXT = "A closer ratio of men to women should be striven for by parties when lists are drawn up."

# BASE_TEXT = "A law is needed to prevent companies from relocating their production abroad."
# VARIANT_TEXT = "Relative authorities need to pass a law to prevent companies from relocating their production abroad."

# BASE_TEXT = "A heritage tax one's wealth should be introduced."
# VARIANT_TEXT = "Relative authorities should introduce a heritage tax one's wealth."

# BASE_TEXT = "Compulsory vaccination of children should be introduced based on the Swiss vaccination plan."
# VARIANT_TEXT = "The government should introduce compulsory vaccination of children based on the Swiss vaccination plan."

# No. 20
# BASE_TEXT = "Automatic facial recognition should be banned in public spaces."
# VARIANT_TEXT = "The government should ban automatic facial recognition in public spaces."

# BASE_TEXT = "Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border."
# VARIANT_TEXT = "The Schengen agreement with the EU should be terminated by Switzerland and more security checks directly on the border should be reintroduced."

BASE_TEXT = "Switzerland should terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons."
VARIANT_TEXT = "The Bilateral Agreements with the EU should be terminated by Switzerland and a free trade agreement without the free movement of persons should be sought."


print_top_layers = 34
TEMP_FOR_PROBS = 1.0
EPS = 1e-9


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
def attn_head_ablation_23(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ratio: float = 0.0,
    all_positions: bool = False,
    head: int = 0
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
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    heads: List[int],
    ratio: float = 0.0,
    all_positions: bool = False
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
            
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, r))
        return results

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

    layer_ablate_results, layer_best_probs = sweep_layer_ablate(ratio=0.0)
    print(f"[Ablate-BLOCK Best-Patched-Probs] {layer_best_probs}")
    print("-" * 60)
    print_top("[Ablate-BLOCK ratio=0.0] top layers", layer_ablate_results)


    def sweep_attn_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        for l in range(n_layers):
            with attn_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio, pos_strategy="last"):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            results.append((l, r))
        return results, best_patched_probs

    ablate0_results, best_probs = sweep_attn_ablate(ratio=0.0)
    print(f"[Ablate-ATTN Best-Patched-Probs] {best_probs}")
    print("-" * 60)
    print_top("[Ablate-ATTN ratio=0.0] top layers", ablate0_results)


    def sweep_mlp_ablate(ratio: float = 0.0):
        results = []
        best_r = 0.0
        best_patched_probs = None
        for l in range(n_layers):
            with mlp_ablation_context(model, enc_corrupt, layers_to_edit=[l], ratio=ratio):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

                r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
                if r > best_r:
                    best_r = r
                    best_patched_probs = patched_probs

            results.append((l, r))
        return results, best_patched_probs

    mlp_ablate0_results, mlp_best_probs = sweep_mlp_ablate(ratio=0.0)
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
    _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)
