import os, re, json, math, sys, random, nltk
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def ensure_nltk():
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt")
    try: nltk.data.find("tokenizers/punkt_tab")
    except LookupError: nltk.download("punkt_tab")
    try: nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError: nltk.download("averaged_perceptron_tagger_eng")
    try: nltk.data.find("corpora/wordnet")
    except LookupError: nltk.download("wordnet")
ensure_nltk()

INPUT_CSV   = "data/original_statements.csv"
OUTPUT_CSV  = "data/synonyms_variants_1.csv"
SWN_PATH    = "SentiWordNet_3.0.0.txt"
WN_DICT_DIR = None

DELTA_THRESH = 0.2
POS_PRIORITY = ["v", "a", "r", "n"]
TOPK_SENSES  = 1                     # WSD 失败时的后备义项最多看前几个
RANDOM_SEED  = 42

lemmatizer = WordNetLemmatizer()
detok = TreebankWordDetokenizer()

def penn_to_wn(pos_tag:str)->Optional[str]:
    if pos_tag.startswith("J"): return "a"
    if pos_tag.startswith("V"): return "v"
    if pos_tag.startswith("N"): return "n"
    if pos_tag.startswith("R"): return "r"
    return None

def is_proper_noun(pos_tag:str)->bool:
    return pos_tag in ("NNP", "NNPS")

def try_lemminflect(base:str, tag:str)->Optional[str]:
    try:
        forms = getInflection(base, tag=tag)
        if forms:
            return forms[0]
    except Exception:
        pass
    return None

def inflect_to_tag(lemma_lower:str, pos_tag:str)->str:
    """
    尝试把 lemma 变到与原 token 同样的词形（用 Penn tag 指示）。
    先用 lemminflect; 失败则退回简化规则。
    """
    # 1) lemminflect（若可用）
    inflected = try_lemminflect(lemma_lower, pos_tag)
    if inflected:
        return inflected
    return lemma_lower

def match_casing(src:str, tgt:str)->str:
    if src.isupper(): return tgt.upper()
    if src.istitle(): return tgt.title()
    return tgt

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
    if pos == "s": pos = "a"
    ps, ns = SWN_DB.get((pos, int(offset)), (0.0, 0.0))
    return float(ps - ns)

def neighbor_synsets(s:Any)->List[Any]:
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
            rel += list(s.also_sees())
    except Exception:
        pass
    kept, seen = [], set()
    for t in rel:
        p = t.pos()
        if p == "s": p = "a"
        if p != pos:  # 只保留同词性的
            continue
        key = (p, t.offset())
        if key not in seen:
            kept.append(t); seen.add(key)
    return kept

def disambiguate_synset(tokens:List[str], idx:int, wn_pos:str)->Optional[Any]:
    word = tokens[idx]
    context = tokens
    try:
        s = lesk(context, word, pos=wn_pos)
        if s: return s
    except Exception:
        pass
    syns = wn.synsets(word, pos=wn_pos)
    if not syns: return None
    # 回退：在前 TOPK 里，优先 SWN 命中且 |net| 大的
    scored = []
    for s in syns[:TOPK_SENSES]:
        net = swn_net(s.pos(), s.offset())
        scored.append((abs(net), len(s.definition()), s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored else syns[0]

def best_replacement_for_token(tokens:List[str], tags:List[str], idx:int)->Optional[Dict[str,Any]]:
    tok = tokens[idx]
    tag = tags[idx]
    wn_pos = penn_to_wn(tag)
    if not wn_pos: return None
    if wn_pos == "n" and is_proper_noun(tag):  # 排除专有名词
        return None

    lemma = lemmatizer.lemmatize(tok.lower(), wn_pos)
    syn = disambiguate_synset(tokens, idx, wn_pos)
    if not syn: return None

    src_net = swn_net(syn.pos(), syn.offset())
    neighbors = neighbor_synsets(syn)

    best = None
    for t in neighbors:
        t_net = swn_net(t.pos(), t.offset())
        delta = abs(t_net - src_net)
        # 阈值规则：|Δnet|≥阈值 或 极性翻转
        if not (delta >= DELTA_THRESH or (src_net * t_net) < 0):
            continue

        # 候选词仅取单词（不含空格/下划线）
        lemmas = [l.name().replace("_"," ") for l in t.lemmas()]
        for cand in lemmas:
            if " " in cand: 
                continue
            if cand.lower() == lemma.lower():
                continue
            # 词形 & 大小写
            inflected = inflect_to_tag(cand.lower(), tag)
            repl = match_casing(tok, inflected)

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

    # 按优先级收集索引
    pos_buckets: Dict[str, List[int]] = {p: [] for p in POS_PRIORITY}
    for i, tag in enumerate(tags):
        p = penn_to_wn(tag)
        if not p: continue
        if p == "n" and is_proper_noun(tag):
            continue
        pos_buckets.setdefault(p, []).append(i)

    # 依优先级选择Δnet最大的一个替换
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

def apply_replacement(sentence:str, rep:Dict[str,Any])->str:
    tokens = word_tokenize(sentence)
    tokens[rep["idx"]] = rep["tgt_word"]
    return detok.detokenize(tokens)

def rewrite_id_tail_7digits(original_id:str)->str:
    new_id, n = re.subn(r'(\d{7})$', '0000001', str(original_id))
    return new_id if n == 1 else str(original_id)

def detect_columns(df: pd.DataFrame)->Tuple[str,str]:
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("id")
    stmt_col = cols_lower.get("statement")
    if not id_col or not stmt_col:
        raise ValueError(f"Could not find 'ID' and 'statement'.")
    return id_col, stmt_col

def process_csv(input_csv:str, output_csv:str):
    df = pd.read_csv(input_csv)
    id_col, stmt_col = detect_columns(df)
    num_success = 0

    # 版本兼容性快速自检
    probe = wn.synsets("good", pos="a")[:3] + wn.synsets("bad", pos="a")[:3]
    hits = sum(1 for s in probe if (("a" if s.pos() in ("a","s") else s.pos()), s.offset()) in SWN_DB)
    if hits < max(1, len(probe)//2):
        print("[WARN] WordNet offsets 似乎与 SentiWordNet_3.0.0 不完全匹配，请确认使用 WordNet 3.0。")

    out_rows = []
    total = len(df)

    for i, row in df.iterrows():
        orig_id = str(row[id_col])
        base_sentence = str(row[stmt_col])

        print(f"\n[{i+1}/{total}] Processing ID={orig_id}")
        rep = choose_best_for_sentence(base_sentence)

        if rep:
            variant = apply_replacement(base_sentence, rep)
            delta_signed = rep["tgt_net"] - rep["src_net"]
            tag = f"[{rep['src_word']}→{rep['tgt_word']}; Δnet={delta_signed:+.2f}]"
            print(f"  Base    : {base_sentence}")
            print(f"  Replace : {rep['src_word']}  ->  {rep['tgt_word']}  "
                  f"(pos={rep['pos']}, offset_src={rep['src_syn'].offset()}, offset_tgt={rep['tgt_syn'].offset()})")
            print(f"  Scores  : src={rep['src_net']:.2f}, tgt={rep['tgt_net']:.2f}, |Δnet|={abs(delta_signed):.2f}")
            print(f"  Variant : {variant}")
            print(f"  Tag     : {tag}")
            num_success += 1
        else:
            variant = None
            tag = ""  # 没有合格替换则留空
            print(f"  Base    : {base_sentence}")
            print("  No eligible replacement (kept original).")
            print(f"  Variant : {variant}")

        new_id = rewrite_id_tail_7digits(orig_id)
        out_rows.append({id_col: new_id, stmt_col: variant, "DeltaTag": tag})

    out_df = pd.DataFrame(out_rows, columns=[id_col, stmt_col, "DeltaTag"])
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"{num_success}/239 succeed.")
    print(f"\nDone. Wrote {len(out_df)} rows to: {output_csv}")

if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_CSV)
