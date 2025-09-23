# swn_substitution_no_llm.py
# -*- coding: utf-8 -*-
import os, re, json, math, sys, random
from typing import Dict, Any, List, Tuple, Optional

# ===================== CONFIG =====================
INPUT_CSV   = "/mnt/data/original_statements.csv"
OUTPUT_CSV  = "/mnt/data/synonyms_variants.csv"
SWN_PATH    = "/mnt/data/SentiWordNet_3.0.0.txt"   # 你的 SentiWordNet 3.0.0 路径
WN_DICT_DIR = None  # 如果你有本机 WordNet-3.0/dict 目录，填这里；否则用 NLTK 自带

DELTA_THRESH = 0.2
POS_PRIORITY = ["v", "a", "r", "n"]  # 动词>形容词>副词>名词（名词会排除专有名词）
TOPK_SENSES  = 5                     # WSD 失败时的后备义项最多看前几个
RANDOM_SEED  = 42

# =============== NLTK 初始化 =======================
import nltk
def ensure_nltk():
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt")
    try: nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError: nltk.download("averaged_perceptron_tagger")
    try: nltk.data.find("corpora/wordnet")
    except LookupError: nltk.download("wordnet")
    try: nltk.data.find("corpora/omw-1.4")
    except LookupError: nltk.download("omw-1.4")
ensure_nltk()

from nltk.corpus import wordnet as wn_default
from nltk.wsd import lesk as nltk_lesk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus.reader.wordnet import WordNetCorpusReader

if WN_DICT_DIR:
    wn = WordNetCorpusReader(WN_DICT_DIR, None)
else:
    wn = wn_default

lemmatizer = WordNetLemmatizer()
detok = TreebankWordDetokenizer()
random.seed(RANDOM_SEED)

# ============== SentiWordNet 3.0.0 读取 ============
def load_swn_txt(path:str)->Dict[Tuple[str,int], Tuple[float,float]]:
    db = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6: continue
            pos = parts[0].strip()
            try:
                off = int(parts[1])
                ps  = float(parts[2]); ns = float(parts[3])
            except ValueError:
                continue
            db[(pos, off)] = (ps, ns)
    return db

SWN_DB = load_swn_txt(SWN_PATH)

def swn_net(pos:str, offset:int)->float:
    """返回 net=pos-neg；形容词卫星's'按'a'查。查不到返回0.0"""
    if pos == "s": pos = "a"
    ps, ns = SWN_DB.get((pos, int(offset)), (0.0, 0.0))
    return float(ps - ns)

# ============== 工具：POS & 词形 ====================
def penn_to_wn(pos_tag:str)->Optional[str]:
    """Penn→WordNet POS"""
    if pos_tag.startswith("J"): return "a"
    if pos_tag.startswith("V"): return "v"
    if pos_tag.startswith("N"): return "n"
    if pos_tag.startswith("R"): return "r"
    return None

def is_proper_noun(pos_tag:str)->bool:
    return pos_tag in ("NNP", "NNPS")

def simple_verb_inflect(base:str, tag:str)->str:
    """非常简化的动词词形变化（不处理不规则动词）"""
    w = base
    vowels = "aeiou"
    if tag == "VB" or tag == "VBP":
        return w
    if tag == "VBZ":
        if re.search(r"(s|x|z|ch|sh)$", w): return w + "es"
        if re.search(r"[^aeiou]y$", w): return w[:-1] + "ies"
        return w + "s"
    if tag in ("VBD","VBN"):
        if w.endswith("e"): return w + "d"
        if re.search(r"[^aeiou]y$", w): return w[:-1] + "ied"
        return w + "ed"
    if tag == "VBG":
        if w.endswith("ie"): return w[:-2] + "ying"
        if w.endswith("e") and not w.endswith("ee"): return w[:-1] + "ing"
        return w + "ing"
    return w

def simple_noun_inflect(base:str, tag:str)->str:
    """非常简化的名词复数（不处理不规则形式）"""
    if tag == "NNS":
        if re.search(r"(s|x|z|ch|sh)$", base): return base + "es"
        if re.search(r"[^aeiou]y$", base): return base[:-1] + "ies"
        return base + "s"
    return base

def match_casing(src:str, tgt:str)->str:
    if src.isupper(): return tgt.upper()
    if src.istitle(): return tgt.title()
    return tgt

# ============== 近义关系收集（保语义） ==============
def neighbor_synsets(s:Any)->List[Any]:
    """根据词性挑选“近义关系”的 synset 候选"""
    rel = []
    pos = s.pos()
    if pos == "s": pos = "a"
    try:
        if pos == "a":
            rel += list(s.similar_tos())
            rel += list(s.also_sees())
        elif pos == "v":
            rel += list(s.verb_groups())
            rel += list(s.also_sees())
        elif pos == "r":
            rel += list(s.also_sees())
        elif pos == "n":
            # 名词保守：偶有 also_see，可用；其他关系容易改事实，谨慎不用
            rel += list(s.also_sees())
    except Exception:
        pass
    # 只保留同词性的（把's'视为'a'）
    kept = []
    for t in rel:
        p = t.pos()
        if p == "s": p = "a"
        if p == pos:
            kept.append(t)
    # 去重
    uniq = []
    seen = set()
    for t in kept:
        key = (t.pos(), t.offset())
        if key not in seen:
            uniq.append(t)
            seen.add(key)
    return uniq

# ============== 词义消歧（先用 Lesk） ===============
def disambiguate_synset(tokens:List[str], idx:int, wn_pos:str)->Optional[Any]:
    """
    先用 NLTK Lesk（基于定义重叠的朴素方法），失败则回退到第一个常见义项。
    """
    word = tokens[idx]
    context = tokens
    try:
        s = nltk_lesk(context, word, pos=wn_pos)
        if s: return s
    except Exception:
        pass
    # 回退：取前 TOPK_SENSES 里 SWN 命中且 gloss 最短的（粗略启发）
    syns = wn.synsets(word, pos=wn_pos)
    if not syns: return None
    scored = []
    for s in syns[:TOPK_SENSES]:
        net = swn_net(s.pos(), s.offset())
        scored.append((abs(net), len(s.definition()), s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored else syns[0]

# ============== 候选生成与筛选 ======================
def best_replacement_for_token(tokens:List[str], tags:List[str], idx:int)->Optional[Dict[str,Any]]:
    tok = tokens[idx]
    tag = tags[idx]
    wn_pos = penn_to_wn(tag)
    if not wn_pos: return None
    if wn_pos == "n" and is_proper_noun(tag):  # 排除专有名词
        return None

    # 词元化（按 POS）
    lemma = lemmatizer.lemmatize(tok.lower(), wn_pos)
    syn = disambiguate_synset(tokens, idx, wn_pos)
    if not syn: return None

    src_net = swn_net(syn.pos(), syn.offset())
    # 收集近义关系的synset
    neighbors = neighbor_synsets(syn)

    best = None  # (delta, target_word, target_synset, target_net)
    for t in neighbors:
        t_net = swn_net(t.pos(), t.offset())
        delta = abs(t_net - src_net)
        if not (delta >= DELTA_THRESH or (src_net * t_net) <= 0):
            continue
        # 只考虑**单词**候选（跳过多词+带下划线）
        lemmas = [l.name().replace("_"," ") for l in t.lemmas()]
        for cand in lemmas:
            if " " in cand: 
                continue
            if cand.lower() == lemma.lower():
                continue
            # 词形匹配
            repl = cand.lower()
            if wn_pos == "v":
                repl = simple_verb_inflect(repl, tag)
            elif wn_pos == "n":
                repl = simple_noun_inflect(repl, tag)
            # 大小写匹配
            repl = match_casing(tok, repl)
            # 记录最佳
            score = delta
            if (best is None) or (score > best["delta"]):
                best = {
                    "src_word": tok, "src_syn": syn, "src_net": src_net,
                    "tgt_word": repl, "tgt_lemma": cand, "tgt_syn": t, "tgt_net": t_net,
                    "delta": score, "idx": idx, "pos": wn_pos, "tag": tag
                }
    return best

def choose_best_for_sentence(sentence:str)->Optional[Dict[str,Any]]:
    tokens = word_tokenize(sentence)
    tags = [t for _, t in nltk.pos_tag(tokens)]

    # 按优先级分桶索引
    pos_buckets: Dict[str, List[int]] = {p: [] for p in POS_PRIORITY}
    for i, tag in enumerate(tags):
        p = penn_to_wn(tag)
        if not p: continue
        if p == "n" and is_proper_noun(tag):  # 名词里剔除专有名词
            continue
        pos_buckets.setdefault(p, []).append(i)

    # 依次尝试每个POS组，选该组里Δnet最大的替换
    for p in POS_PRIORITY:
        indices = pos_buckets.get(p, [])
        group_best = None
        for idx in indices:
            cand = best_replacement_for_token(tokens, tags, idx)
            if not cand: continue
            if (group_best is None) or (cand["delta"] > group_best["delta"]):
                group_best = cand
        if group_best:
            return group_best
    return None

# ============== 句子替换与输出 ======================
def apply_replacement(sentence:str, rep:Dict[str,Any])->str:
    tokens = word_tokenize(sentence)
    tokens[rep["idx"]] = rep["tgt_word"]
    return detok.detokenize(tokens)

def rewrite_id_tail_7digits(original_id:str)->str:
    new_id, n = re.subn(r'(\d{7})$', '0000001', str(original_id))
    return new_id if n == 1 else str(original_id)

# ============== CSV 批处理 =========================
import pandas as pd

def detect_columns(df: pd.DataFrame)->Tuple[str,str]:
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id")
    stmt_col = cols_lower.get("statement")
    if not id_col or not stmt_col:
        raise ValueError(f"无法在 CSV 里找到 'ID' 与 'statement' 两列。实际列名：{list(df.columns)}")
    return id_col, stmt_col

def process_csv(input_csv:str, output_csv:str):
    df = pd.read_csv(input_csv)
    id_col, stmt_col = detect_columns(df)

    out_rows = []
    total = len(df)

    # 版本兼容性快速自检（避免 3.1 dict 搭 3.0 SWN）
    probe = wn.synsets("good", pos="a")[:3] + wn.synsets("bad", pos="a")[:3]
    hits = sum(1 for s in probe if (("a" if s.pos() in ("a","s") else s.pos()), s.offset()) in SWN_DB)
    if hits < max(1, len(probe)//2):
        print("[WARN] WordNet offsets似乎与SentiWordNet_3.0.0不完全匹配，请确认使用WordNet 3.0。")

    for i, row in df.iterrows():
        orig_id = str(row[id_col])
        base_sentence = str(row[stmt_col])

        print(f"[{i+1}/{total}] Processing ID={orig_id}")
        rep = choose_best_for_sentence(base_sentence)

        if rep:
            variant = apply_replacement(base_sentence, rep)
            delta_signed = rep["tgt_net"] - rep["src_net"]
            print(f"  Replace: {rep['src_word']} -> {rep['tgt_word']}  "
                  f"(Δnet={delta_signed:+.2f}; src={rep['src_net']:.2f}, tgt={rep['tgt_net']:.2f}; "
                  f"pos={rep['pos']}, offset_src={rep['src_syn'].offset()}, offset_tgt={rep['tgt_syn'].offset()})")
        else:
            variant = base_sentence
            print("  No eligible replacement (kept original).")

        new_id = rewrite_id_tail_7digits(orig_id)
        out_rows.append({id_col: new_id, stmt_col: variant})

    out_df = pd.DataFrame(out_rows, columns=[id_col, stmt_col])
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Done. Wrote {len(out_df)} rows to: {output_csv}")

# ================== MAIN ==========================
if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_CSV)
