import re
import pandas as pd

original_csv = "data/original_statements.csv"
variant_csv = "data/it-clefts_CI_12B.csv"
original_not_sig = "data/significance/original_12B_not_significant.csv"
variant_not_sig = "data/significance/it-clefts_12B_not_significant.csv"

def extract_base_id(id_str):
    s = str(id_str)
    m = re.match(r'[a-z]{2}_[0-9]{1,2}', s)
    return m.group()

def load_file(path):
    df = pd.read_csv(path)
    df = df.copy()
    df["base_id"] = df["ID"].apply(extract_base_id)
    return df[["base_id", "ID"]]

df_original = load_file(original_csv).set_index("base_id")
df_original_not_sig = load_file(original_not_sig).set_index("base_id")
df_variant = load_file(variant_csv).set_index("base_id")
df_variant_not_sig = load_file(variant_not_sig).set_index("base_id")

common_ov = df_original.index.intersection(df_variant.index)          # 1) 交集
exclude = df_original_not_sig.index.union(df_variant_not_sig.index)   # 2) 要剔除的集合（并集）
remain = common_ov.difference(exclude)                                # 3) 剩下的

print(len(remain))
