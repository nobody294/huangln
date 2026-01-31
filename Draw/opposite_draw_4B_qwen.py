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
MODEL_NAME = "Qwen/Qwen3-4B"  # or "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

PAIRS: List[Tuple[str, str]] = [
    ("Childcare should be free for all parents for at least three days a week.",
     "Child care should remain paying for all parents at least three days a week."),
    ("Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists.",
     "A tax per kilometer driven should not replace the existing tax on car ownership for motorists."),
    ("Houses should be built on land currently used for agriculture.",
     "Land now used for agriculture should only be used for agriculture."),
    ("Both the purchase and sale of soft drugs by coffee shops should be legalized.",
     "Both purchase and sale of soft drugs by coffee shops should remain illegal."),
    ("People who consider their lives complete should be able to receive assistance with suicide.",
     "People who find their lives complete should be denied assisted suicide."),
    ("New residential areas should consist of at least 40 percent green space.",
     "New housing developments should consist of up to 40 percent social housing."),
    ("The growth of Islam is a threat to Spain's security.",
     "The growth of Islam is harmless to Spain's security."),
    ("Climate change is solely attributable to human action.",
     "Climate change is only attributable to natural action."),
    ("The future Spanish government should increase irrigated agricultural areas by means of large water transfers.",
     "The future Spanish government should reduce agricultural irrigated areas through large water transfers."),
    ("Negotiating with pro-independence supporters weakens the State.",
     "Negotiating with the independentistas strengthens the State."),
    ("Young people over the age of 16 are to be allowed to vote in Bundestag elections.",
     "Only adults (18 and over) should be allowed to vote in federal elections."),
    ("The traditional family of father, mother and children is to be promoted more strongly than other living arrangements.",
     "Other cohabiting couples are to be promoted more than the traditional family of father, mother and children."),
    ("Facial recognition software should be allowed to be used for video surveillance in public places.",
     "The use of facial recognition software for video surveillance in public places should be prohibited."),
    ("The statutory minimum wage is to be increased to at least 12 euros by 2022 at the latest.",
     "The statutory minimum wage is to be reduced by 2022 at the latest."),
    ("The result of any nationwide referendum should be binding regardless of turnout.",
     "The result of any nationwide referendum should be binding depending on turnout."),
    ("The share of defense spending in Poland's GDP should be further increased.",
     "The share of defense spending in Poland's GDP should be reduced."),
    ("The powers of local governments should be increased at the expense of the central government.",
     "The powers of local governments should be reduced at the expense of the central government."),
    ("Christian values should be the basis of state social policy.",
     "State social policy should be independent of Christian values."),
    ("The President of the Hungarian Republic should be directly elected.",
     "The President of the Hungarian Republic should be elected by the Parliament."),
    ("A legal framework for primary elections should be provided.",
     "A civilian regulatory framework for primaries should be provided."),
    ("Internet access should be free for all.",
     "Internet should be free for all."),
    ("A heritage tax one's wealth should be introduced.",
     "A wealth tax on great wealth should be cut."),
    ("Health care should be managed only by the state and not by private individuals.",
     "Health care should be managed only by private individuals and not by the state. "),
    ("The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services). ",
     "The federal government should be removed the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."),
    ("Same-sex couples should have the same rights as heterosexual couples in all areas. ",
     "Same-sex couples should have more rights as heterosexual couples in all areas."),
    ("A minimum wage of CHF 4,000 for all full-time employees should be introduced. ",
     "A minimum wage of CHF 4,000 for all full-time employees should be disregarded. "),
    ("The Swiss mobile network should be equipped throughout the country with the latest technology (currently 5G standard). ",
     "The Swiss mobile network should be kept the same throughout the country with the latest technology (currently 5G standard)."),
]

# PAIRS: List[Tuple[str, str]] = [
#     ("Organizers of events should be able to request a vaccination certificate upon entry.",
#      "It should be impossible for event organizers to be able to request a vaccination certificate."),

#     ("The Netherlands should exit the European Union (EU).",
#      "The Netherlands must remain in the European Union (EU)."),

#     ("Households with two partners, one of whom works, should receive the same tax benefits as households with two working partners.",
#      "Households with two partners of which one works should receive less tax benefit as households with two working partners."),

#     ("The Dutch government should apologize for the historical slave trade.",
#      "The slave trade of the past is not the responsibility of the current Dutch government."),

#     ("There should be fewer options for community service sentences instead of prison sentences.",
#      "There should be more opportunities to impose community service instead of prison sentences."),

#     ("An increase in minimum wages should no longer automatically result in an increase in welfare benefits.",
#      "Increasing minimum wages should automatically lead to increases in welfare benefits."),

#     ("A middle school should be established so that students make a choice between vocational education, general secondary education, or pre-university education at a later age.",
#      "The arrival of a middle school is unnecessary, because students are now old enough when they have the choice between vmbo, havo or vwo."),

#     ("People should always have the choice of whether to wear a face mask.",
#      "Not the people themselves, but other agencies should choose whether or not to wear a mouth guard."),

#     ("Limiting rights and freedoms is necessary to combat organized crime.",
#      "It is necessary to expand rights and freedoms to combat organized crime."),

#     ("The state should take measures to redistribute wealth from the rich to the poor.",
#      "The state must take measures to increase the gap between rich and poor."),

#     ("The government must increase spending on public health care, even if this means increasing taxes.",
#      "The government should decrease spending on public health care so as not to increase taxes."),

#     ("Taxes on fossil fuels must be raised to finance the Green Transition.",
#      "Taxes on fossil fuels should be reduced and the Ecological Transition should be ignored."),

#     ("To better defend Spain's interests in Europe we must recover more sovereignty.",
#      "To better defend Spain's interests in Europe, we must cede sovereignty."),

#     ("Spanish government should promote the strengthening of NATO in Europe.",
#      "The Spanish government should promote the weakening of NATO in Europe."),

#     ("The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum.",
#      "The worst way to solve the conflict in Catalonia is to allow its citizens to vote on their future in a referendum."),

#     ("Spain's territorial decentralization must be deepened.",
#      "The centralization of power in Spain must be deepened."),

#     ("The right to self-determination must be recognized by the Constitution.",
#      "The right of self-determination must be ignored by the Constitution."),

#     ("Spain should be more tolerant with illegal migration.",
#      "Spain should be more intolerant of illegal immigration."),

#     ("Housing prices must be regulated to ensure access for all people.",
#      "Housing prices should be left to the free market."),

#     ("A general speed limit is to apply on all highways.",
#      "Unlimited speed should be allowed on all highways."),

#     ("The ability of landlords to increase housing rents is to be more strictly limited by law.",
#      "Landlords should be allowed to increase rents without legal restrictions."),

#     ("The right of recognized refugees to join their families is to be abolished.",
#      "The right of recognized refugees to family reunification is to be extended."),

#     ("Donations from companies to political parties should continue to be permitted.",
#      "Donations from companies to political parties should be prohibited."),

#     ("In Germany, it should generally be possible to have a second citizenship in addition to the German one.",
#      "In Germany, it should only be possible to have a single citizenship."),

#     ("Federal authorities are to take linguistic account of different gender identities in their publications.",
#      "Federal authorities should not use gender-neutral language in their publications."),

#     ("Female civil servants are to be allowed to wear headscarves while on duty.",
#      "Female civil servants should generally be banned from wearing headscarves on duty."),

#     ("The controlled sale of cannabis is to be generally permitted.",
#      "The controlled sale of cannabis should be prohibited."),

#     ("Germany is to leave the European Union.",
#      "Germany should remain a member of the European Union."),

#     ("Organic agriculture should be promoted more strongly than conventional agriculture.",
#      "Conventional agriculture should be promoted more than organic farming."),

#     ("Islamic associations are to be able to be recognized by the state as religious communities.",
#      "Islamic associations should be rejected by the state as religious communities."),

#     ("The debt brake in the Basic Law is to be retained.",
#      "The debt brake in the Basic Law is to be lifted."),

#     ("The state should provide a free nursery place for every child.",
#      "The state should refrain from providing free nursery places for all children."),

#     ("The state should build low-rent apartments for rent.",
#      "The state should refrain from building low-income rental housing."),

#     ("The independence of the judiciary from parliament and the government should be strengthened.",
#      "Parliamentary and government control over the judiciary should be strengthened."),

#     ("Poland should move away from coal mining no later than 2040.",
#      "Poland should continue coal mining beyond 2040."),

#     ("Poland should have grain imports from Ukraine blocked.",
#      "Poland should support grain imports from Ukraine."),

#     ("The powers of the secret services to track the activities of citizens on the Internet should be limited.",
#      "The powers of the secret services to track citizens' activities on the Internet should be increased."),

#     ("Hungary should decide by referendum whether to remain part of the EU.",
#      "Hungary should decide to remain part of the EU without consulting the electorate."),

#     ("Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities).",
#      "Gender identity is formed independently of environmental influences (e.g. media content, sensitising activities)."),

#     ("Hungary should join the European Public Prosecutor's Office.",
#      "Hungary should withdraw from the European Public Prosecutor's Office."),

#     ("Stricter regulation of interception software (e.g. Pegasus) is needed (e.g. subject to judicial authorisation).",
#      "Lighter regulation of interception software (e.g. Pegasus) is needed."),

#     ("Education spending should be increased to at least the OECD average of 5.2 per cent (GDP).",
#      "Spending on education is sufficient."),

#     ("Only men and women should be allowed to marry.",
#      "Same-sex couples should be allowed to marry."),

#     ("The state should take targeted measures to promote equal participation of fathers and mothers in child-rearing.",
#      "The state takes targeted measures to prevent fathers and mothers from sharing equally in child-rearing."),

#     ("The Hungarian government should ratify the Istanbul Convention, which combats violence against women and domestic violence.",
#      "The Hungarian government should reject the ratification of the Istanbul Convention, which combats violence against women and domestic violence."),

#     ("Comprehensive public procurement reform is needed (e.g. opening up large-scale centralised public procurement to smaller firms).",
#      "Comprehensive public procurement reform is unnecessary (e.g. opening up large-scale centralised public procurement to smaller companies)."),

#     ("Increase the contribution of the wealthier to the public purse (abolition of the one-band tax).",
#      "The more wealthy should contribute less to the public burden (abolition of the one-band tax)."),

#     ("A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation.",
#      "A price freeze on some basic foodstuffs (e.g. chicken tails, milk) is ineffective in combating inflation."),

#     ("State regulation of the rental housing market is not necessary.",
#      "Public regulation of the rental housing market is required."),

#     ("The use of medical cannabis should be legalised in Hungary.",
#      "Make the use of medical cannabis illegal in Hungary."),

#     ("Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner-take-all compensation, extension of postal voting) is needed.",
#      "Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner's compensation, extension of postal voting) is unnecessary."),

#     ("In larger cities, car traffic should be limited through various measures (P+R parking, construction of cycle paths, improvement of public transport).",
#      "In larger cities, it is unnecessary to restrict car traffic by various measures (P+R parking, building cycle paths, improving public transport)."),

#     ("The redevelopment of urban green spaces (e.g. the Liget project in Budapest) needs a broad social dialogue.",
#      "In the case of the redevelopment of urban green areas (e.g. the Liget project in Budapest), a broad social dialogue is unjustified."),

#     ("An independent ministry for the environment is needed.",
#      "A separate environment ministry is unnecessary."),

#     ("An animal rights commissioner should be introduced.",
#      "The introduction of an animal rights commissioner is unnecessary."),

#     ("European integration is all in all a positive process.",
#      "European integration is an all-negative process."),

#     ("The European Union should have a common foreign policy.",
#      "The European Union should cancel the common foreign policy."),

#     ("Migrant landings must be stopped, even by extreme means.",
#      "Migrant landings must continue, ceasing to resort to extreme means."),

#     ("Children, born in Italy to foreign citizens and who have completed schooling should be granted Italian citizenship (ius scholae).",
#      "Children, born in Italy to foreign nationals and who have completed schooling should have their Italian citizenship (ius scholae) denied."),

#     ("More civil rights should be granted to homosexual, bisexual, transgender (LGBT+) people.",
#      "Civil rights should be limited to homosexual, bisexual, transgender (LGBT+) people."),

#     ("Recreational use of marijuana/cannabis should be allowed.",
#      "Recreational use of marijuana/cannabis should be prohibited."),

#     ("An hourly minimum wage should be introduced.",
#      "The hourly minimum wage should be ignored."),

#     ("The construction of Major Works is a priority for Italy.",
#      "The construction of Major Works is irrelevant to Italy."),

#     ("Drilling is necessary to find more energy resources.",
#      "Drilling is irrelevant to finding more energy resources."),

#     ("Regasifiers are necessary infrastructure for Italy.",
#      "Regasifiers are irrelevant infrastructure for Italy."),

#     ("Italy should keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO).",
#      "Italy should cancel its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."),

#     ("Separation of careers between judges and prosecutors should be introduced.",
#      "Career separation between judges and prosecutors is irrelevant."),

#     ("Direct election of the President of the Republic should be introduced.",
#      "Direct election of the President of the Republic is irrelevant."),

#     ("Paid parental leave should be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave. ",
#      "Paid parental leave should be reduced under today's 14 weeks of maternity leave and two weeks of paternity leave."),

#     ("According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in regular classes. ",
#      "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in separate classes. "),

#     ("The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). ",
#      "The state should neglect equal educational opportunities (e.g., regarding subsidized remedial courses for students from low-income families). "),

#     ("More qualified workers from non-EU/EFTA countries should be allowed to work in Switzerland (increase third-country quota).",
#      "More qualified workers from non-EU/EFTA countries should be forbidden to work in Switzerland (decrease third-country quota)."),

#     ("Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. ",
#      "Foreign nationals who have lived in Switzerland for at least ten years should be refused the right to vote and stand for election at the municipal level."),

#     ("Cannabis use should be legalized. ",
#      "Cannabis use should be kept ilegal."),

#     ("Doctors should be allowed to administer direct active euthanasia. ",
#      "Doctors should be fobidden to administer direct active euthanasia."),

#     ("The differences between cantons with high and low financial capacity should be further reduced through fiscal equalization. ",
#      "The differences between cantons with high and low financial capacity should be further increased through fiscal equalization."),

#     ("There should be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). ",
#      "There should be laxer regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "),

#     ("Private households should be free to choose their electricity supplier (complete liberalization of the electricity market). ",
#      "Private households should be limited in choosing their electricity supplier (full regulation of the electricity market)."),

#     ("There should be stricter controls on equal pay for women and men. ",
#      "There should be laxer controls on equal pay for women and men."),

#     ("The state should guarantee a comprehensive public service offering also in rural regions. ",
#      "The state should ignore a comprehensive public service offering also in rural regions."),

#     ("Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be introduced. ",
#      "Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be disregarded."),

#     ("The protection regulations for large predators (lynx, wolf, bear) should be relaxed. ",
#      "The protection regulations for large predators (lynx, wolf, bear) should be made stricter. "),

#     ("Direct payments should only be granted to farmers with proof of ecological performance. ",
#      "Direct payments should be granted to all farmers without requiring proof of ecological performance."),

#     ("There should be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas). ",
#      "There should be laxer animal welfare regulations for livestock (e.g. only temporary access to outdoor areas)."),

#     ("30% of Switzerland's land area should be dedicated to preserving biodiversity?. ",
#      "Switzerland should ignore the allocation of any specific percentage of its land area to preserving biodiversity."),

#     ("There should be a ban on single-use plastic and non-recyclable plastics. ",
#      "There should be an incentive to use single-use plastic and non-recyclable plastics."),

#     ("There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). ",
#      "The government should ignore measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "),

#     ("There should be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). ",
#      "There should be a laxer regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). "),

#     ("The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). ",
#      "The Federal Council should be forbidden to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "),

#     ("Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border. ",
#      "Switzerland should keep the Schengen agreement with the EU. There's no need for more security checks directly on the border."),

#     ("Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. ",
#      "Companies should be ignore whether their subsidiaries and suppliers operating abroad comply with social and environmental standards."),
# ]


RULE_NAME = "Opposite"

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
    p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
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
    # print(f"\n[Figure 1] Activation patching profiles for rule: {RULE_NAME}")
    # patch_profiles = []
    # for i, (b, v) in enumerate(PAIRS, 1):
    #     print(f"  Patching pair {i}/{len(PAIRS)}")
    #     patch_profiles.append(patching_profile_for_pair(model, tokenizer, b, v))
    # patch_stats = aggregate_profiles(patch_profiles)

    # fig1 = plot_layerwise_subplot(
    #     title=f"Activation patching\n{RULE_NAME} (unflip pairs n={patch_stats['n_pairs']})",
    #     stats=patch_stats,
    #     ylabel="Normalized restoration score",
    # )
    # fig1.savefig("opposite_patching_one_rule_qwen.png", dpi=200)
    # print("Saved: opposite_patching_one_rule.png")

    # ---------------------------
    # Figure 2: Ablation curves (ratio=0.0)
    # ---------------------------
    print(f"\n[Figure 2] Ablation (ratio=0.0) profiles for rule: {RULE_NAME}")
    ab_profiles = []
    for i, (b, v) in enumerate(PAIRS, 1):
        print(f"  Ablation pair {i}/{len(PAIRS)}")
        ab_profiles.append(ablation_profile_for_pair(model, tokenizer, b, v, ratio=0.0))
    ab_stats = aggregate_profiles(ab_profiles)

    fig2 = plot_layerwise_subplot(
        title=f"Inference-time masking (ratio=0.0)\n{RULE_NAME} (flip pairs n={ab_stats['n_pairs']})",
        stats=ab_stats,
        ylabel="Normalized restoration score",
    )
    fig2.savefig("opposite_ablation_one_rule_qwen.png", dpi=200)
    print("Saved: opposite_ablation_one_rule.png")

    plt.show()


if __name__ == "__main__":
    main()
