import os
import re
import csv
import random
import numpy as np
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3Attention,
)

# ============================================================
# 配置
# ============================================================

model_name = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

TEMP_FOR_PROBS = 1.0
EPS = 1e-9

# ===========================
# 通用工具
# ===========================

def set_global_determinism(seed: int = 42, single_thread: bool = True):
    """可选：设置随机种子，方便复现实验。"""
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

def get_input_device(model: Gemma3ForConditionalGeneration):
    """在 device_map='auto' 情况下更稳的拿到设备。"""
    try:
        return model.model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device

def get_decoder_layers(model: Gemma3ForConditionalGeneration):
    """返回 (层序号, 模块名, Gemma3DecoderLayer 模块) 列表。"""
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

# ===========================
# EncodedChat & 编码相关
# ===========================

@dataclass
class EncodedChat:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_pos: int
    digit_ids: List[int]


def build_user_prompt(statement: str) -> str:
    """把 statement 包装成打 1–7 分的用户指令。"""
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
    """把 system + user prompt 编成一段 chat，answer_pos=最后一个 token。"""
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

    # 把 1–7 映射到 tokenizer 的单 token id
    digit_ids: List[int] = []
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

# ===========================
# 前向 + 只取 1–7 的概率
# ===========================

@torch.no_grad()
def forward_logits_only(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat
) -> torch.Tensor:
    out = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        output_hidden_states=False,
        return_dict=True
    )
    # 只取 answer_pos 的 logits
    logits = out.logits[:, enc.answer_pos, :].squeeze(0)
    return logits


def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def digit_probs_from_logits_full(
    logits_full: torch.Tensor, enc: EncodedChat, temperature: float = 1.0
) -> torch.Tensor:
    digits = digit_logit_slice(logits_full, enc.digit_ids)
    return torch.softmax(digits / temperature, dim=-1)

# ===========================
# 距离 & 恢复度指标
# ===========================

def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    一维 Earth Mover's Distance (Wasserstein-1)，
    这里实现成 CDF 差的 L1。
    """
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(
    dist_fn,
    p_clean: torch.Tensor,
    p_corrupt: torch.Tensor,
    p_patched: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    R = 1 - dist(clean, patched) / dist(clean, corrupt)
    dist(clean, corrupt) 很小时返回 NaN。
    """
    d0 = dist_fn(p_clean, p_corrupt)
    dp = dist_fn(p_clean, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float('nan')), R)

# ===========================
# 注意力 ablation（第 23 层）
# ===========================

@contextlib.contextmanager
def attn_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layers_to_edit: List[int],
    ratio: float = 0.0,
    pos_strategy: str = "fixed"
):
    """
    在给定的层里，把 Gemma3Attention 的输出缩放为 ratio。
    - ratio = 0.0: 完全 ablate
    - pos_strategy = "last": 只在序列最后一个 token ablate
    - pos_strategy = "fixed": 在 enc.answer_pos ablate（这里两者等价）
    """
    hooks = []

    def make_hook(layer_idx: int):
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
            if isinstance(sub, Gemma3Attention):
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
    """
    只在第 23 层把 attention 输出乘上 ratio（默认 0：完全 ablate）。
    """
    with attn_ablation_context(
        model,
        enc,
        layers_to_edit=[layer_to_edit],
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
def attn_head_scaling_up_23_multiple_heads(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    ratio: float = 2.0,
    all_positions: bool = False
):
    with attn_head_ablation_context(
        model,
        enc,
        layers_to_edit=[23],
        heads_to_edit=[0, 2, 4],
        ratio=ratio,
        all_positions=all_positions
    ):
        yield

# ============================================================
# 下面是 CSV 构造部分（ID 配对 + 筛选）
# ============================================================

# ID 形如: 两个小写字母_一或两位数字_七位数字
_ID_RE = re.compile(r'^([a-z]{2}_[0-9]{1,2})_([0-9]{7})$')


def _extract_prefix(id_str: str) -> Optional[str]:
    """
    从 'ab_3_1234567' 提取前缀 'ab_3'。
    """
    if not id_str:
        return None
    m = _ID_RE.match(id_str.strip())
    return m.group(1) if m else None


def _is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _load_id_stmt_map(csv_path: str):
    """
    读取形如 original_statements / it-clefts_variants：
    要求有列: ID, statement
    返回:
      - pref2pair: {前缀: (完整ID, 句子)}
      - stats: 一些统计信息
    """
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
            stmt = (row.get('statement') or '').strip()

            pref = _extract_prefix(id_raw)
            if not pref:
                stats["bad_id"] += 1
                continue

            if pref in seen_prefix:
                stats["dup_prefix"] += 1
                continue

            if not _is_nonempty_text(stmt):
                stats["empty_stmt"] += 1
                seen_prefix.add(pref)
                continue

            pref2pair[pref] = (id_raw, stmt)
            seen_prefix.add(pref)

    return pref2pair, stats


def _load_flip_prefix_set(csv_path: str):
    """
    只负责从文件里收集“前缀集合”，用于过滤：
    接受任何有 ID 列的 CSV。
    """
    flip_prefixes = set()
    stats = {
        "rows": 0,
        "bad_id": 0,
        "dup_prefix": 0,
    }
    seen_prefix = set()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'ID' not in reader.fieldnames:
            raise ValueError(f"{csv_path} 必须包含列: ID")
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


def build_filtered_pairs(
    base_csv_path: str,
    variant_csv_path: str,
    flip_csv_path: Optional[str] = None,
    base_non_significant_csv_path: Optional[str] = None,
    variant_non_significant_csv_path: Optional[str] = None,
    keep_order_by_base: bool = True,
    verbose: bool = True
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """
    按你的规则构造“干净句子对”：
      1) 从 base / variant 中找到共有前缀（ID 前两段）
      2) 去掉出现在 flip / base_non_sig / variant_non_sig 中的所有前缀
      3) 返回：
         - shared_prefixes: List[str]
         - base_texts: List[str]
         - variant_texts: List[str]
         - report: 一些统计信息
    """

    base_map, base_stats = _load_id_stmt_map(base_csv_path)
    var_map, var_stats = _load_id_stmt_map(variant_csv_path)

    base_keys = set(base_map.keys())
    var_keys = set(var_map.keys())

    common_before_flip = base_keys & var_keys

    if flip_csv_path:
        flip_prefixes, flip_stats = _load_flip_prefix_set(flip_csv_path)
    else:
        flip_prefixes, flip_stats = set(), {"rows": 0, "bad_id": 0, "dup_prefix": 0}

    if base_non_significant_csv_path:
        base_non_significant_prefixes, base_non_stats = _load_flip_prefix_set(base_non_significant_csv_path)
    else:
        base_non_significant_prefixes, base_non_stats = set(), {"rows": 0, "bad_id": 0, "dup_prefix": 0}

    if variant_non_significant_csv_path:
        v_non_significant_prefixes, v_non_stats = _load_flip_prefix_set(variant_non_significant_csv_path)
    else:
        v_non_significant_prefixes, v_non_stats = set(), {"rows": 0, "bad_id": 0, "dup_prefix": 0}

    # 真正保留的前缀
    common = common_before_flip - flip_prefixes - base_non_significant_prefixes - v_non_significant_prefixes

    # 是否按 base 文件原顺序来排列前缀
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

    base_sentences = [base_map[p][1] for p in prefix_list]
    variant_sentences = [var_map[p][1] for p in prefix_list]

    blocked_by_flip = len(common_before_flip & flip_prefixes)
    blocked_by_base_non_significant = len(common_before_flip & base_non_significant_prefixes)
    blocked_by_variant_non_significant = len(common_before_flip & v_non_significant_prefixes)

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
        "blocked_by_flip": blocked_by_flip,
        "blocked_by_base_non_significant": blocked_by_base_non_significant,
        "blocked_by_variant_non_significant": blocked_by_variant_non_significant,
    }

    if verbose:
        print(f"[CSV] base rows={report['base_rows']} "
              f"(bad_id={report['bad_id_base']}, empty_stmt={report['empty_stmt_base']}, "
              f"dup={report['dup_prefix_base']})")
        print(f"[CSV] variant rows={report['variant_rows']} "
              f"(bad_id={report['bad_id_variant']}, empty_stmt={report['empty_stmt_variant']}, "
              f"dup={report['dup_prefix_variant']})")
        if flip_csv_path:
            print(f"[CSV] flip rows={report['flip_rows']}; blocked_by_flip={report['blocked_by_flip']}")
        print(f"[CSV] paired(after all filters)={report['paired']}, "
              f"only_in_base_after_filter={report['only_in_base_after_filter']}, "
              f"only_in_variant_after_filter={report['only_in_variant_after_filter']}")
        print(f"[CSV] blocked_by_base_non_significant={report['blocked_by_base_non_significant']}")
        print(f"[CSV] blocked_by_variant_non_significant={report['blocked_by_variant_non_significant']}")
        for p in prefix_list[:3]:
            print(f"[CSV] sample pair prefix={p} | "
                  f"base_id={base_map[p][0]} | variant_id={var_map[p][0]}")

    return prefix_list, base_sentences, variant_sentences, report

# ============================================================
# 主流程：生成句子对 CSV + 计算 ablate-23 的恢复分数
# ============================================================

def main():
    # ---------- 1. 数据路径（根据你的实际路径改一下） ----------

    BASE_CSV_PATH = "data/original_statements.csv"
    VARIANT_CSV_PATH = "data/it-clefts_variants.csv"
    FLIP_CSV_PATH = "data/flip rate/it-clefts_flip_4B.csv"
    BASE_NON_SIG_PATH = "data/significance/original_4B_not_significant.csv"
    VAR_NON_SIG_PATH = "data/significance/it-clefts_4B_not_significant.csv"

    # 输出文件名
    PAIRS_CSV_PATH = "data/paired_filtered_statements.csv"
    PAIRS_WITH_R_CSV_PATH = "data/paired_filtered_statements_with_r.csv"

    # ---------- 2. 构造干净的句子对 ----------
    prefixes, base_texts, variant_texts, rep = build_filtered_pairs(
        BASE_CSV_PATH,
        VARIANT_CSV_PATH,
        flip_csv_path=FLIP_CSV_PATH,
        base_non_significant_csv_path=BASE_NON_SIG_PATH,
        variant_non_significant_csv_path=VAR_NON_SIG_PATH,
        keep_order_by_base=True,
        verbose=True,
    )

    assert len(prefixes) == len(base_texts) == len(variant_texts)
    num_pairs = len(prefixes)
    print(f"\n[Info] 最终句子对数量: {num_pairs}")

    # ---------- 3. 写一个三列的 CSV（只包含 ID + 两个句子） ----------
    with open(PAIRS_CSV_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shared_id", "original_statement", "it_cleft_statement"])
        for sid, b, v in zip(prefixes, base_texts, variant_texts):
            writer.writerow([sid, b, v])

    print(f"[Info] 已写出三列句子对 CSV: {PAIRS_CSV_PATH}")

    if num_pairs == 0:
        print("[Warn] 没有句子对，后续不计算恢复分数。")
        return

    # ---------- 4. 载入模型与 processor ----------
    print("\n[Info] 正在加载 Gemma-3-4B-it 模型（可能需要一点时间）...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    ).eval()

    device = get_input_device(model)
    print(f"[Info] 模型主设备: {device}")

    # ---------- 5. 对每个句子对计算 ablate-23 的恢复分数 ----------
    r_scores: List[float] = []

    for idx, (sid, base_text, variant_text) in enumerate(zip(prefixes, base_texts, variant_texts), start=1):
        # 编码 clean / corrupt
        enc_clean = encode_for_next_token(
            processor, model, SYSTEM_PROMPT, build_user_prompt(base_text)
        )
        enc_corrupt = encode_for_next_token(
            processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text)
        )

        with torch.no_grad():
            logits_clean = forward_logits_only(model, enc_clean)
            logits_corrupt = forward_logits_only(model, enc_corrupt)
            clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
            corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

        # 对 corrupt 句子做第 23 层 attention ablation
        # with attn_ablation_23(model, enc_corrupt, layer_to_edit=23, ratio=0.0):
        #     logits_patched = forward_logits_only(model, enc_corrupt)
        #     patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)

        # r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
        # r_value = float(r.item())
        # r_scores.append(r_value)

        # print(f"[Progress] 已处理 {idx}/{num_pairs} 个句子对，当前 R={r_value:.4f}")
        # print(f"[Clean Probs] {clean_probs}")
        # print(f"[Corrupt Probs] {corrupt_probs}")
        # print(f"[Patched Probs] {patched_probs}")
        # print("-" * 60)

        # 对corrupt句子对做第23层的attention scaling
        with attn_head_scaling_up_23_multiple_heads(model, enc_corrupt, ratio=7.0, all_positions=False):
            logits_patched = forward_logits_only(model, enc_corrupt)
            patched_probs = digit_probs_from_logits_full(logits_patched, enc_clean, TEMP_FOR_PROBS)
        
        r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
        r_value = float(r.item())
        r_scores.append(r_value)

        print(f"[Progress] 已处理 {idx}/{num_pairs} 个句子对，当前 R={r_value:.4f}")
        print(f"[Clean Probs] {clean_probs}")
        print(f"[Corrupt Probs] {corrupt_probs}")
        print(f"[Patched Probs] {patched_probs}")
        print("-" * 60)

    # ---------- 6. 写出带 R 分数的 CSV ----------
    with open(PAIRS_WITH_R_CSV_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shared_id",
            "original_statement",
            "it_cleft_statement",
            "r_w1_restoration_layer23"
        ])
        for sid, b, v, r in zip(prefixes, base_texts, variant_texts, r_scores):
            writer.writerow([sid, b, v, r])

    print(f"\n[Info] 已写出带恢复分数的 CSV: {PAIRS_WITH_R_CSV_PATH}")
    print(f"[Info] R 分数示例（前 5 个）： {r_scores[:5]}")


if __name__ == "__main__":
    # 根据需要可以关掉 deterministic（设成 False）
    set_global_determinism(0, single_thread=True)
    main()
