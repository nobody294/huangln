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
from contextlib import contextmanager

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

_ID_RE = re.compile(r'^([a-z]{2}_[0-9]{1,2})_([0-9]{7})$')

# Example pair for base vs variant (you can change these)
BASE_TEXT = "The government should abolish the ban on face-covering clothing."
VARIANT_TEXT = "It is the ban on face-covering clothing that the government should abolish."
# VARIANT_TEXT = "The ban on face-covering clothing should be abolished by the government."

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

def set_global_determinism(seed: int = 42, single_thread: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # 或 ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if single_thread:
        os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; torch.set_num_threads(1)
# ---------------------------
# Utilities / Model Introspection
# ---------------------------

# =========================
# ActAdd (Activation Addition) utilities
# =========================
def _get_layer_by_index(model: Gemma3ForConditionalGeneration, layer_idx: int) -> Gemma3DecoderLayer:
    for i, _, layer in get_decoder_layers(model):
        if i == layer_idx:
            return layer
    raise IndexError(f"Decoder layer {layer_idx} not found.")

@torch.no_grad()
def _capture_resid_pre_for_ids(
    model: Gemma3ForConditionalGeneration,
    input_ids: torch.Tensor,           # [1, seq]
    attention_mask: Optional[torch.Tensor],
    layer_idx: int
) -> torch.Tensor:
    """
    抽取指定层 **输入**（resid_pre）的整个序列激活: [seq, hidden] (CPU, float32)
    """
    holder = {}

    def pre_hook(module, inputs):
        # inputs: (hidden_states, attention_mask, position_ids, ...)
        h = inputs[0]                     # [B, seq, hidden]
        holder["resid_pre"] = h.detach().clone().squeeze(0).to(torch.float32).cpu()
        return None  # 不改输入

    layer = _get_layer_by_index(model, layer_idx)
    h = layer.register_forward_pre_hook(pre_hook)

    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    finally:
        h.remove()

    if "resid_pre" not in holder:
        raise RuntimeError("Failed to capture resid_pre; check hooks/model version.")
    return holder["resid_pre"]  # [seq, hidden] CPU float32

def _tokenize_raw(tok, text: str, max_len: int = 1024):
    enc = tok(text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len)
    return enc["input_ids"]

def _pad_right_same_len(ids_a: torch.Tensor, ids_b: torch.Tensor, pad_id: int):
    """
    右侧补齐到同长度（论文 Algorithm 1 第 1 行）。
    """
    # 以 ids_a 的 device 为基准，必要时把 ids_b 也搬过来
    if ids_b.device != ids_a.device:
        ids_b = ids_b.to(ids_a.device)

    La = ids_a.size(-1)
    Lb = ids_b.size(-1)
    L  = max(La, Lb)

    if La < L:
        # 与 ids_a 同 dtype/device 的填充
        pad_a = ids_a.new_full((1, L - La), pad_id)
        ids_a = torch.cat([ids_a, pad_a], dim=-1)
    if Lb < L:
        pad_b = ids_b.new_full((1, L - Lb), pad_id)
        ids_b = torch.cat([ids_b, pad_b], dim=-1)

    return ids_a, ids_b

@torch.no_grad()
def build_actadd_vector_seq(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    p_plus: str, p_minus: str,
    layer_idx: int,
    max_len: int = 1024,
    pad_token_strategy: str = "eos"  # "eos" | "space"
) -> torch.Tensor:
    """
    返回 steering 序列 h_A^l: [seq, hidden] (CPU float32)
    论文做法：p+ 与 p- 右补齐同长 → 抽 resid_pre → 做差（Algorithm 1）。
    """
    tok = processor.tokenizer
    dev = get_input_device(model)

    ids_p = _tokenize_raw(tok, p_plus, max_len=max_len).to(dev)
    ids_m = _tokenize_raw(tok, p_minus, max_len=max_len).to(dev)

    # 选择 pad token：优先 eos，其次单空格（若是单 token）
    if pad_token_strategy == "eos" and tok.eos_token_id is not None:
        pad_id = tok.eos_token_id
    else:
        space_ids = tok.encode(" ", add_special_tokens=False)
        pad_id = space_ids[0] if len(space_ids) == 1 else (tok.eos_token_id or ids_m[0, -1].item())

    ids_p, ids_m = _pad_right_same_len(ids_p, ids_m, pad_id)

    attn_p = torch.ones_like(ids_p, device=ids_p.device)
    attn_m = torch.ones_like(ids_m, device=ids_m.device)

    H_p = _capture_resid_pre_for_ids(model, ids_p, attn_p, layer_idx)  # [L, H] cpu f32
    H_m = _capture_resid_pre_for_ids(model, ids_m, attn_m, layer_idx)  # [L, H] cpu f32
    return (H_p - H_m)  # steering 序列

# ======== 多对 (p+, p-) 平均 steering ========
@torch.no_grad()
def build_actadd_vector_seq_from_pairs(
    model, processor,
    pairs: List[Tuple[str, str]],    # [(p_plus, p_minus), ...]
    layer_idx: int,
    max_len: int = 1024,
    pad_token_strategy: str = "eos",
) -> List[torch.Tensor]:
    """
    对每一对 (p+, p-) 生成一条 steering 序列 h_A^l，返回列表 [ [Li,H], ... ] (CPU float32)。
    """
    seqs = []
    for (p_plus, p_minus) in pairs:
        hA = build_actadd_vector_seq(
            model, processor, p_plus, p_minus,
            layer_idx=layer_idx, max_len=max_len, pad_token_strategy=pad_token_strategy
        )  # [Li, H] cpu f32
        seqs.append(hA)
    return seqs

def _right_pad_zero(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把若干 [Li,H] 右补齐为 [N, Lmax, H]，返回:
      padded: [N, Lmax, H]
      count_mask: [Lmax, 1]  —— 每个位置有多少条序列“有值”
    """
    assert len(seqs) > 0
    H = seqs[0].shape[1]
    Lmax = max(s.shape[0] for s in seqs)
    N = len(seqs)
    padded = torch.zeros((N, Lmax, H), dtype=torch.float32)
    count = torch.zeros((Lmax, 1), dtype=torch.float32)
    for i, s in enumerate(seqs):
        Li = s.shape[0]
        padded[i, :Li, :] = s
        count[:Li, 0] += 1.0
    return padded, count

@torch.no_grad()
def mean_steering_from_pairs(
    model, processor,
    pairs: List[Tuple[str, str]],
    layer_idx: int,
    max_len: int = 1024,
    pad_token_strategy: str = "eos",
) -> torch.Tensor:
    """
    生成“平均 steering 序列” \bar h_A^l: [Lmax, H]。
    只对“有值的位置”平均；无样本覆盖的位置保持为 0。
    """
    seqs = build_actadd_vector_seq_from_pairs(
        model, processor, pairs, layer_idx=layer_idx, max_len=max_len, pad_token_strategy=pad_token_strategy
    )
    padded, count = _right_pad_zero(seqs)   # [N,Lmax,H], [Lmax,1]
    summed = padded.sum(dim=0)              # [Lmax,H]
    mean = summed / count.clamp_min(1.0)    # 逐位置平均
    mean[count.squeeze(1) == 0] = 0.0       # 无覆盖的位置归零（保险）
    return mean.contiguous()                # [Lmax, H] cpu f32


@contextmanager
def actadd_injection_context(
    model: Gemma3ForConditionalGeneration,
    hA_seq: torch.Tensor,    # [seq_s, hidden] CPU float32
    layer_idx: int,
    coeff: float = 1.0,
    align_pos: int = 1       # 论文默认 a=1，即“front alignment”
):
    """
    在层 l 的 **输入 residual** 处做 S <- S + c * hA 对齐注入（Algorithm 1）。
    仅对包裹的 forward 生效；退出后自动移除 hook。
    """
    seq_s, H = hA_seq.shape
    hA_seq = hA_seq.contiguous()

    def pre_hook(module, inputs):
        hidden = inputs[0]  # [B, seq, H]
        B, L, Hd = hidden.shape
        assert Hd == H, f"Hidden size mismatch: {Hd} vs {H}"
        start = max(0, align_pos)  # 一般 a=1（即从 token 1 开始）
        end = min(L, start + seq_s)
        if end <= start:
            return None

        add = hA_seq[:(end-start), :].to(hidden.device, dtype=hidden.dtype) * coeff
        new_h = hidden.clone()
        new_h[:, start:end, :] = new_h[:, start:end, :] + add.unsqueeze(0)
        # 返回替换后的 inputs tuple
        return (new_h, ) + tuple(inputs[1:])

    layer = _get_layer_by_index(model, layer_idx)
    h = layer.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        h.remove()


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

# ======== 自动对齐 a 到“用户文本起点” ========
def _find_subseq(haystack: List[int], needle: List[int]) -> int:
    """在 token 序列里找子序列起点；找不到返回 -1。"""
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1
    # 朴素搜索足够稳定
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

@torch.no_grad()
def compute_align_pos_user_start(
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
) -> int:
    """
    把 a 设置为“完整 chat 模板里 user_prompt 这段文本的起始 token 下标”。
    找不到就退回 1。
    """
    # token 化“完整 chat”
    messages = [
        {"role": "system", "content": [{"type":"text","text": system_prompt}]},
        {"role": "user",   "content": [{"type":"text","text": user_prompt}]},
    ]
    enc_full = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True
    )
    full_ids = enc_full["input_ids"][0].tolist()

    # 单独 token 化 user_prompt（不加 special）
    tok = processor.tokenizer
    user_ids = tok(user_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

    idx = _find_subseq(full_ids, user_ids)
    return 1 if idx < 0 else idx


def run_actadd_once(
    train_base: List[str],
    train_variant: List[str],
    eval_pair: Tuple[str, str],
    # 取一对 (p+, p-) 来构造 steering；你也可以改成从多个对里做平均
    steering_idx: int = 0,
    steering_indices: Optional[List[int]] = None,
    candidate_layers: Optional[List[int]] = None,
    coeff_grid = (0.2, 0.5, 1.0, 1.5, 2.0),
    align_pos: int | str = 1,
    use_raw_for_steering: bool = True,
):
    """
    论文一致的 ActAdd 实验：
      1) 用 (p+, p-) 在层 l 抽 resid_pre 做差 → hA^l（可多层反复算）
      2) 对 variant 前向时在层输入加 c*hA^l（a=align_pos）
      3) 选出在 W1/JS 还原度最好且 PPL 不超阈的 (l,c)
    """
    assert len(train_base) > 0 and len(train_variant) > 0
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    ).eval()

    # 选择一对 (p+, p-) ；此处我们用“base 句子为 +；variant 为 -”
    p_plus   = train_base[steering_idx]
    p_minus  = train_variant[steering_idx]

    # 候选层：默认扫“中间到偏后”的若干层（论文经验中间层更稳）
    layers = get_decoder_layers(model)
    nL = len(layers)
    if candidate_layers is None:
        candidate_layers = list(range(15, 34))

    # 评估基线
    enc_clean   = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(eval_pair[0]))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(eval_pair[1]))
    with torch.no_grad():
        logits_clean   = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
    
    # 计算 “自动 a” 的位置（对 variant 的 user prompt）
    if isinstance(align_pos, str) and align_pos.lower() == "auto_user_start":
        variant_user_prompt = build_user_prompt(eval_pair[1])
        a_auto = compute_align_pos_user_start(processor, SYSTEM_PROMPT, variant_user_prompt)
    else:
        a_auto = int(align_pos)


    clean_probs   = digit_probs_from_logits_full(logits_clean,   enc_clean,   TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    base_nll, base_ppl, _ = evaluate_wikitext_ppl(
        model, processor,
        dataset_config="wikitext-2-raw-v1", split="test",
        block_size=None, stride=None, max_texts=200
    )
    print(f"[WikiText] baseline: PPL={base_ppl:.3f}")

    best = None
    best_meta = None

    if steering_indices is not None and len(steering_indices) > 0:
        pairs = [(train_base[i], train_variant[i]) for i in steering_indices]
    else:
        pairs = [(train_base[steering_idx], train_variant[steering_idx])]

    for l in candidate_layers:
        # 1) 为该层构造 steering 序列
        if len(pairs) == 1:
            hA = build_actadd_vector_seq(model, processor, pairs[0][0], pairs[0][1], layer_idx=l,
                                         max_len=1024, pad_token_strategy="eos")
        else:
            hA = mean_steering_from_pairs(model, processor, pairs, layer_idx=l,
                                          max_len=1024, pad_token_strategy="eos")

        for c in coeff_grid:
            with actadd_injection_context(model, hA, layer_idx=l, coeff=c, align_pos=a_auto):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs  = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
                # 通用能力（PPL）
                patched_nll, patched_ppl, _ = evaluate_wikitext_ppl(
                    model, processor,
                    dataset_config="wikitext-2-raw-v1", split="test",
                    block_size=None, stride=None, max_texts=200
                )

            r_w1 = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs).item()
            ppl_up = (patched_ppl / base_ppl - 1.0) * 100.0
            ok = (ppl_up <= 20.0)  # 例：不超过 +20%（你可以改严一点）

            print(f"[ActAdd] layer={l:02d} c={c:.2f}  R_W1={r_w1:.3f}  PPL={patched_ppl:.3f} (Δ={ppl_up:.1f}%)")

            if ok and (best is None or (not math.isnan(r_w1) and r_w1 > best)):
                best = r_w1
                best_meta = (l, c, patched_ppl)

    if best_meta is None:
        print("No (layer, coeff) met the PPL constraint.")
        return

    l_best, c_best, ppl_best = best_meta
    delta_pct = (ppl_best / base_ppl - 1.0) * 100.0
    print(f"[ActAdd] BEST layer={l_best} c={c_best:.2f}  restoration={best:.3f}")
    print(f"[WikiText] baseline PPL={base_ppl:.3f} → patched PPL={ppl_best:.3f} (Δ={delta_pct:.1f}%)")

    # 终端打印对比分布
    with actadd_injection_context(model, build_actadd_vector_seq(model, processor, p_plus, p_minus, layer_idx=l_best),
                                  layer_idx=l_best, coeff=c_best, align_pos=align_pos):
        logits_best = forward_logits_only(model, enc_corrupt)
        probs_best  = digit_probs_from_logits_full(logits_best, enc_clean, TEMP_FOR_PROBS)
    print(f"[Clean probs]   {clean_probs}")
    print(f"[Corrupt probs] {corrupt_probs}")
    print(f"[Patched probs] {probs_best}")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    torch.set_grad_enabled(True)

    set_global_determinism(0, single_thread=True)

    # print("=== Baseline diagnostics: activation patching / ablation ===")
    # _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)

    print("\n=== ActAdd (paper-faithful) ===")
    BASE_CSV_PATH    = "data/original_statements.csv"
    VARIANT_CSV_PATH = "data/it-clefts_variants.csv"
    FLIP_CSV_PATH    = "data/flip rate/it-clefts_flip_4B.csv"

    train_base, train_variant, rep = build_train_lists_from_csv(
        BASE_CSV_PATH, VARIANT_CSV_PATH, FLIP_CSV_PATH,
        keep_order_by_base=False, verbose=True
    )
    if len(train_base) == 0:
        raise RuntimeError("从 CSV 没配出任何成对样本。")

    eval_pair = (BASE_TEXT, VARIANT_TEXT)

    # 用第 0 对 (base, variant) 构造 (p+, p-) 的 steering；你也可换成别的 index
    run_actadd_once(
        train_base, train_variant, eval_pair,
        steering_indices=range(0, 201),
        candidate_layers=None,            # 默认扫一段中高层
        coeff_grid=(0.2, 0.5, 1.0, 1.5, 2.0),
        align_pos="auto_user_start"
    )
