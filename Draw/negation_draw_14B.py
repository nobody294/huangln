import os
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# Try importing Qwen3 internal module classes (used for isinstance checks).
# If your transformers is too old, this import may fail; we fall back to class-name checks.
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP
    _HAVE_QWEN3_CLASSES = True
except Exception:
    Qwen3DecoderLayer = None
    Qwen3Attention = None
    Qwen3MLP = None
    _HAVE_QWEN3_CLASSES = False


# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "Qwen/Qwen3-14B"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

PAIRS: List[Tuple[str, str]] = [
    ("There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). ",
     "There should not be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "),

    ("The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). ",
     "The Federal Council should not be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "),

    ("Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. ",
     "Companies should not be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards."),

    ("As part of the reform of the BVG (occupational pension plan), pensions are to be reduced (lowering the minimum conversion rate from 6.8% to 6%). ",
     "As part of the reform of the BVG (occupational pension plan), pensions are not to be reduced (lowering the minimum conversion rate from 6.8% to 6%). "),

    ("The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). ",
     "The state should not be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). "),

    ("Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. ",
     "Foreign nationals who have lived in Switzerland for at least ten years should not be granted the right to vote and stand for election at the municipal level."),

    ("Same-sex couples should have the same rights as heterosexual couples in all areas. ",
     "Same-sex couples should not have the same rights as heterosexual couples in all areas."),

    ("The state should guarantee a comprehensive public service offering also in rural regions. ",
     "The state should not guarantee a comprehensive public service offering also in rural regions."),

    ("30% of Switzerland's land area should be dedicated to preserving biodiversity?. ",
     "30% of Switzerland's land area should not be dedicated to preserving biodiversity. "),

    ("Federal authorities are to take linguistic account of different gender identities in their publications.",
     "Federal authorities should not use different gender identities in their publications."),

    ("Female civil servants are to be allowed to wear headscarves while on duty.",
     "Female civil servants should not be allowed to wear headscarves on duty."),

    ("The state lists of the parties for the elections to the German Bundestag are to have to be filled alternately by women and men.",
     "The state lists of the parties for the elections to the German Bundestag should not have to be filled alternately by women and men."),

    ("The right of recognized refugees to join their families is to be abolished.",
     "The right of recognized refugees to family reunification should not be abolished."),

    ("Taxes on fossil fuels must be raised to finance the Green Transition.",
     "Taxes on fossil fuels should not be raised to finance the Ecological Transition."),

    ("To better defend Spain's interests in Europe we must recover more sovereignty.",
     "In order to better defend Spain's interests in Europe, we should not recover more sovereignty."),

    ("The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum.",
     "The best way to solve the conflict in Catalonia is that its citizens cannot vote on their future in a referendum."),

    ("The government must increase spending on public health care, even if this means increasing taxes.",
     "The government should not increase spending on the public health system even if this means increasing taxes."),

    ("Education spending should be increased to at least the OECD average of 5.2 per cent (GDP).",
     "Spending on education should not be increased to the OECD average of 5.2 per cent (of GDP)."),

    ("Only men and women should be allowed to marry.",
     "Marriages should not be exclusively between men and women."),

    ("Comprehensive public procurement reform is needed (e.g. opening up large-scale centralised public procurement to smaller firms).",
     "There is no need for comprehensive public procurement reform (e.g. opening up large-scale centralised public procurement to smaller firms)."),

    ("Public employment helps people re-enter the labour market.",
     "Public works do not help people to re-enter the labour market."),

    ("A legal framework for primary elections should be provided.",
     "There is no need to provide a legal framework for primary elections."),

    ("Polluting companies should be taxed more heavily.",
     "Polluting companies should not be subject to higher taxes."),

    ("In larger cities, car traffic should be limited through various measures (P+R parking, construction of cycle paths, improvement of public transport).",
     "In larger cities, there is no need to restrict car traffic through various measures (P+R parking, building cycle paths, improving public transport)."),

    ("The redevelopment of urban green spaces (e.g. the Liget project in Budapest) needs a broad social dialogue.",
     "The redevelopment of urban green areas (e.g. the Liget project in Budapest) does not require a broad social dialogue."),

    ("Stricter regulation of interception software (e.g. Pegasus) is needed (e.g. subject to judicial authorisation).",
     "There is no need for stricter regulation of interception software (e.g. Pegasus) (e.g. subject to judicial authorisation)."),

    ("Italy should keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO).",
     "Italy should not keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."),

    ("Restrictions on personal freedom and privacy are acceptable to deal with health emergencies such as Covid-19.",
     "Restrictions on personal freedom and privacy are not acceptable for dealing with health emergencies such as Covid-19."),

    ("Children, born in Italy to foreign citizens and who have completed schooling should be granted Italian citizenship (ius scholae).",
     "Children, born in Italy to foreign nationals and who have completed school, should not be granted Italian citizenship (ius scholae)."),

    ("More civil rights should be granted to homosexual, bisexual, transgender (LGBT+) people.",
     "Homosexual, bisexual, transgender (LGBT+) people should not be granted more civil rights."),

    ("The Dutch government should apologize for the historical slave trade.",
     "The Dutch government should not apologize for the slave trade in the past."),

    ("New residential areas should consist of at least 40 percent green space.",
     "New housing developments should not consist of at least 40 percent social housing."),

    ("The independence of the judiciary from parliament and the government should be strengthened.",
     "The independence of the judiciary from parliament and government should not be strengthened."),

    ("Poland should have grain imports from Ukraine blocked.",
     "Poland should not lead to the blocking of grain imports from Ukraine."),

    ("Social transfers should be increased to reduce the effects of inflation on citizens.",
     "Social transfers should not be increased to limit the effects of inflation on citizens."),

    ("The state should build low-rent apartments for rent.",
     "The state should not build low-income rental housing."),

    ("Early retirement should be introduced for those who have worked a certain number of years, regardless of their age.",
     "There should be no early retirement for those who have worked a certain number of years, regardless of their age."),
]


RULE_NAME = "Negation"

TEMP_FOR_PROBS = 1.0
EPS = 1e-12


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
def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _is_instance_or_name(mod, cls, name: str) -> bool:
    if cls is not None:
        return isinstance(mod, cls)
    return mod.__class__.__name__ == name


def get_decoder_layers(model):
    layers = []
    for name, mod in model.named_modules():
        if _is_instance_or_name(mod, Qwen3DecoderLayer, "Qwen3DecoderLayer"):
            layers.append((len(layers), name, mod))
    if not layers:
        # A more direct access for Qwen3ForCausalLM: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for i, layer in enumerate(model.model.layers):
                layers.append((i, f"model.layers.{i}", layer))
    if not layers:
        raise RuntimeError("No Qwen3DecoderLayer found. Check transformers version/model class.")
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


def _apply_chat_template_safely(tokenizer, messages) -> str:
    """
    Qwen3 Instruct 系列可能支持 enable_thinking 参数；为保证“第一个生成token就是数字”，优先关闭 thinking。
    如果当前 tokenizer 不支持该参数，就自动退化为普通调用。
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def encode_for_next_token(
    tokenizer: AutoTokenizer,
    model,
    system_prompt: str,
    user_prompt: str
) -> EncodedChat:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = _apply_chat_template_safely(tokenizer, messages)
    enc = tokenizer([text], return_tensors="pt")

    dev = get_input_device(model)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)

    answer_pos = input_ids.shape[-1] - 1

    digit_ids = []
    for d in range(1, 8):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
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


def flip_probs_1_to_7(p: torch.Tensor) -> torch.Tensor:
    idx = torch.tensor([6, 5, 4, 3, 2, 1, 0], device=p.device)
    return p.index_select(dim=-1, index=idx)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12) -> torch.Tensor:
    # p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_clean, p_corrupt)
    dp = dist_fn(p_clean, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float('nan')), R)


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


def collect_clean_cache(model, enc_clean: EncodedChat) -> CleanCache:
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
            if _is_instance_or_name(sub, Qwen3Attention, "Qwen3Attention"):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif _is_instance_or_name(sub, Qwen3MLP, "Qwen3MLP"):
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
            if _is_instance_or_name(sub, Qwen3Attention, "Qwen3Attention"):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif _is_instance_or_name(sub, Qwen3MLP, "Qwen3MLP"):
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
    model,
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
    model,
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
            if _is_instance_or_name(sub, Qwen3Attention, "Qwen3Attention") and i in layers_to_edit:
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
            if _is_instance_or_name(sub, Qwen3MLP, "Qwen3MLP") and i in layers_to_edit:
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Compute profiles for one pair
# ---------------------------
def patching_profile_for_pair(model, tokenizer, base_text, variant_text):
    enc_clean = encode_for_next_token(tokenizer, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(tokenizer, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    clean_cache = collect_clean_cache(model, enc_clean)
    n_layers = len(get_decoder_layers(model))

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

    return {
        "block": sweep("block"),
        "attn": sweep("attn"),
        "mlp": sweep("mlp"),
        "n_layers": n_layers,
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
    }


def ablation_profile_for_pair(model, tokenizer, base_text, variant_text, ratio: float = 0.0):
    enc_clean = encode_for_next_token(tokenizer, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(tokenizer, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

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


# ---------------------------
# Aggregation
# ---------------------------
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


# ---------------------------
# Main
# ---------------------------
def main():
    set_global_determinism(0, single_thread=True)
    torch.set_grad_enabled(False)

    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    # ---------------------------
    # Figure 1: Activation patching curves
    # ---------------------------
    print(f"\n[Figure 1] Activation patching profiles for rule: {RULE_NAME}")
    patch_profiles = []
    for i, (b, v) in enumerate(PAIRS, 1):
        print(f"  Patching pair {i}/{len(PAIRS)}")
        patch_profiles.append(patching_profile_for_pair(model, tokenizer, b, v))
    patch_stats = aggregate_profiles(patch_profiles)

    fig1 = plot_layerwise_subplot(
        title=f"Activation patching\n{RULE_NAME} (unflip pairs n={patch_stats['n_pairs']})",
        stats=patch_stats,
        ylabel="Normalized restoration score",
    )
    fig1.savefig("negation_patching_one_rule_14B.png", dpi=200)
    print("Saved: negation_patching_one_rule.png")

    # ---------------------------
    # Figure 2: Ablation curves (ratio=0.0)
    # ---------------------------
    # print(f"\n[Figure 2] Ablation (ratio=0.0) profiles for rule: {RULE_NAME}")
    # ab_profiles = []
    # for i, (b, v) in enumerate(PAIRS, 1):
    #     print(f"  Ablation pair {i}/{len(PAIRS)}")
    #     ab_profiles.append(ablation_profile_for_pair(model, tokenizer, b, v, ratio=0.0))
    # ab_stats = aggregate_profiles(ab_profiles)

    # fig2 = plot_layerwise_subplot(
    #     title=f"Inference-time masking (ratio=0.0)\n{RULE_NAME} (flip pairs n={ab_stats['n_pairs']})",
    #     stats=ab_stats,
    #     ylabel="Normalized restoration score",
    # )
    # fig2.savefig("negation_ablation_one_rule_qwen.png", dpi=200)
    # print("Saved: negation_ablation_one_rule.png")

    plt.show()


if __name__ == "__main__":
    main()
