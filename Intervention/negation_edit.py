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

MMLU_SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. "
    "Answer correctly and output exactly one capital letter: A, B, C, or D."
)

# CUDA_VISIBLE_DEVICES=1,2 python Intervention/negation_edit.py

# Example pair for base vs variant (you can change these)
# BASE_TEXT = "There should be an additional tax on purchasing meat."
# VARIANT_TEXT = "There should be a no additional tax on buying meat."

# BASE_TEXT = "The national government, rather than provinces and municipalities, should decide where new residential areas are built."
# VARIANT_TEXT = "The national government should not decide where new residential areas are built instead of provinces and municipalities."

# BASE_TEXT = "Houses should be built on land currently used for agriculture."
# VARIANT_TEXT = "No housing should be built on land now used for agriculture."

# BASE_TEXT = "There should be fewer options for community service sentences instead of prison sentences."
# VARIANT_TEXT = "There should not be fewer opportunities to impose community service instead of prison sentences."

# BASE_TEXT = "People who consider their lives complete should be able to receive assistance with suicide."
# VARIANT_TEXT = "People who consider their lives complete should not be able to get help with suicide."

# BASE_TEXT = "An increase in minimum wages should no longer automatically result in an increase in welfare benefits."
# VARIANT_TEXT = "Raising minimum wages should still automatically increase welfare payments."

# BASE_TEXT = "A middle school should be established so that students make a choice between vocational education, general secondary education, or pre-university education at a later age."
# VARIANT_TEXT = "There should be no middle school, so that pupils do not have to choose between vmbo, havo or vwo at a later age."

# BASE_TEXT = "Limiting rights and freedoms is necessary to combat organized crime."
# VARIANT_TEXT = "It is not necessary to limit rights and freedoms to combat organized crime."

# BASE_TEXT = "The growth of Islam is a threat to Spain's security."
# VARIANT_TEXT = "The growth of Islam is not a threat to Spain's security."

# No. 10
# BASE_TEXT = "The efficiency of public services improves when they are privatized."
# VARIANT_TEXT = "Efficiency in the provision of public services does not improve when they are privatized."

# BASE_TEXT = "It should be easier for companies to fire workers."
# VARIANT_TEXT = "It should not be easier for companies to lay off workers."

# BASE_TEXT = "Climate change is solely attributable to human action."
# VARIANT_TEXT = "Climate change is not solely attributable to human action."

# BASE_TEXT = "The future Spanish government should increase irrigated agricultural areas by means of large water transfers."
# VARIANT_TEXT = "The future Spanish government should not increase agricultural irrigated areas through large water transfers."

# BASE_TEXT = "Negotiating with pro-independence supporters weakens the State."
# VARIANT_TEXT = "Negotiating with the independentistas does not weaken the State."

# BASE_TEXT = "The policies of linguistic immersion in the native language of bilingual Autonomous Communities endanger Spanish."
# VARIANT_TEXT = "The policies of linguistic immersion in the language of the bilingual Autonomous Communities do not endanger Spanish."

# BASE_TEXT = "It is necessary to repeal the Law of Democratic Memory passed during this legislature."
# VARIANT_TEXT = "It is not necessary to repeal the Law of Democratic Memory passed during this legislature."

# BASE_TEXT = "Immigrants should pay for their own health services."
# VARIANT_TEXT = "Immigrants should not have to pay for their health services."

# BASE_TEXT = "Covid-19 vaccines are to continue to be protected by patents."
# VARIANT_TEXT = "Vaccines against Covid-19 should not continue to be protected by patents."

# BASE_TEXT = "The traditional family of father, mother and children is to be promoted more strongly than other living arrangements."
# VARIANT_TEXT = "The traditional family of father, mother and children should not be promoted more than other cohabiting couples."

# No. 20
# BASE_TEXT = "Students should receive BAföG regardless of their parents' income."
# VARIANT_TEXT = "Students should receive BAföG depending on their parents' income."

# BASE_TEXT = "The Nord Stream 2 Baltic Sea pipeline, which transports gas from Russia to Germany, is to be allowed to go into operation as planned."
# VARIANT_TEXT = "The ""Nord Stream 2"" Baltic Sea pipeline, which transports gas from Russia to Germany, should not be allowed to go into operation as planned."

# BASE_TEXT = "The registration of new cars with combustion engines should also be possible in the long term."
# VARIANT_TEXT = "The registration of new cars with combustion engines should no longer be possible in the long term."

# BASE_TEXT = "The state should continue to collect church tax for religious communities."
# VARIANT_TEXT = "The state should not collect church tax for religious communities."

# BASE_TEXT = "Inpatient treatment in hospitals is to continue to be charged on the basis of a flat rate per case."
# VARIANT_TEXT = "Inpatient treatment in hospitals should not be charged at a flat rate per case."

# BASE_TEXT = "A tax is to be levied again on high assets."
# VARIANT_TEXT = "No tax should be levied on high assets."

# BASE_TEXT = "Facial recognition software should be allowed to be used for video surveillance in public places."
# VARIANT_TEXT = "No facial recognition software should be used for video surveillance in public places."

# BASE_TEXT = "Air traffic is to be taxed more heavily."
# VARIANT_TEXT = "Air traffic should not be taxed more heavily."

# BASE_TEXT = "The European Union should have less influence on Polish domestic policy."
# VARIANT_TEXT = "The European Union should not have less influence on Polish domestic policy."

# BASE_TEXT = "The state should finance private visits to specialists if the waiting time at a public facility exceeds three months."
# VARIANT_TEXT = "The state should not finance private visits to specialists if the waiting time at a public facility exceeds three months."

# No. 30
# BASE_TEXT = "The result of any nationwide referendum should be binding regardless of turnout."
# VARIANT_TEXT = "The results of certain nationwide referendums should be binding depending on turnout."

# BASE_TEXT = "Poland should adopt the migrant relocation solutions adopted by the European Union."
# VARIANT_TEXT = "Poland should not adopt the migrant relocation solution adopted by the European Union."

# BASE_TEXT = "The powers of local governments should be increased at the expense of the central government."
# VARIANT_TEXT = "The powers of local governments should not be increased at the expense of the central government."

# BASE_TEXT = "Christian values should be the basis of state social policy."
# VARIANT_TEXT = "Christian values should not be the basis of state social policy."

# BASE_TEXT = "The EU's rule of law mechanism threatens Hungary's sovereignty."
# VARIANT_TEXT = "The EU's rule of law mechanism does not threaten Hungary's sovereignty."

# BASE_TEXT = "Stronger state regulation of the work of NGOs supported by foreign organisations is needed."
# VARIANT_TEXT = "There is no need for stronger state regulation of the work of NGOs supported by foreign organisations."

# BASE_TEXT = "Political influence has been reduced by changing the university model (reorganisation into a trust)."
# VARIANT_TEXT = "The change in the university model (reorganisation into a trust) has not reduced political influence."

# BASE_TEXT = "One effective way to reduce rents is to conclude favourable gas supply contracts with Russia."
# VARIANT_TEXT = "The conclusion of favourable gas supply contracts with Russia is not an effective way of reducing rationing."

# BASE_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation."
# VARIANT_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is not the right way to fight inflation."

# BASE_TEXT = "Migrant landings must be stopped, even by extreme means."
# VARIANT_TEXT = "Migrant landings must not be stopped, even by extreme means."

# No. 40
# BASE_TEXT = "A heritage tax one's wealth should be introduced."
# VARIANT_TEXT = "A wealth tax on great wealth should not be introduced."

# BASE_TEXT = "The citizenship allowance is a measure that should be cancelled."
# VARIANT_TEXT = "The citizenship income is not a measure to be cancelled."

# BASE_TEXT = "Beach concessions to private individuals should be time-limited."
# VARIANT_TEXT = "Beach concessions to private individuals should not be time-limited."

# BASE_TEXT = "Italy should build more incinerators/thermal power plants."
# VARIANT_TEXT = "Italy should not build more incinerators/thermal power plants."

# BASE_TEXT = "Drilling is necessary to find more energy resources."
# VARIANT_TEXT = "Drilling is not necessary to find more energy resources."

# BASE_TEXT = "Taxes collected at the regional level should be fully retained in the region itself."
# VARIANT_TEXT = "Taxes collected at the regional level should not be fully retained in the region itself."

# BASE_TEXT = "Compulsory vaccination of children should be introduced based on the Swiss vaccination plan."
# VARIANT_TEXT = "Compulsory vaccination of children should not be introduced based on the Swiss vaccination plan."

# BASE_TEXT = "Insured persons should contribute more to health care costs (e.g., increase the minimum deductible)."
# VARIANT_TEXT = "Insured persons should not contribute more to health care costs (e.g., increase the minimum deductible)."

# BASE_TEXT = "The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."
# VARIANT_TEXT = "The federal government should not be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."

# BASE_TEXT = "The federal government should raise the requirements for the high school."
# VARIANT_TEXT = "The federal government should not raise the requirements for the high school."

# No. 50
# BASE_TEXT = "A third official gender should be introduced alongside ""female"" and ""male""."
# VARIANT_TEXT = "A third official gender should not be introduced alongside ""female"" and ""male""."

# BASE_TEXT = "To achieve climate targets, incentives and target agreements should be relied on exclusively, rather than bans and restrictions."
# VARIANT_TEXT = "To achieve climate targets, incentives and target agreements should not be relied on exclusively, rather than bans and restrictions."

# BASE_TEXT = "The army's target number of soldiers should expand to at least 120,000."
# VARIANT_TEXT = "The army's target number of soldiers should not expand to at least 120,000."

# BASE_TEXT = "The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine)."
# VARIANT_TEXT = "The Federal Council should not be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine)."

BASE_TEXT = "Switzerland should terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons."
VARIANT_TEXT = "Switzerland should not terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons."


topk_attr = 6          # how many top layers to print/consider in diagnostics
print_top_layers = 34  # how many top layers to print
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
