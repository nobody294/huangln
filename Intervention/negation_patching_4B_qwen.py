import os
import math
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
)

model_name = "Qwen/Qwen3-4B"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# Example pair for base vs variant
# BASE_TEXT = "There should be a ban on single-use plastic and non-recyclable plastics. "
# VARIANT_TEXT = "There should not be a ban on single-use plastic and non-recyclable plastics."

# BASE_TEXT = "There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "
# VARIANT_TEXT = "There should not be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "

# BASE_TEXT = "The Swiss mobile network should be equipped throughout the country with the latest technology (currently 5G standard). "
# VARIANT_TEXT = "The Swiss mobile network should not be equipped throughout the country with the latest technology (currently 5G standard)."

# BASE_TEXT = "The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "
# VARIANT_TEXT = "The Federal Council should not be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine). "

# BASE_TEXT = "Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. "
# VARIANT_TEXT = "Companies should not be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards."

# BASE_TEXT = "Paid parental leave should be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave. "
# VARIANT_TEXT = "Paid parental leave should not be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave."

# BASE_TEXT = "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in regular classes. "
# VARIANT_TEXT = "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should not be taught in regular classes. "

# BASE_TEXT = "The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). "
# VARIANT_TEXT = "The state should not be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). "

# BASE_TEXT = "More qualified workers from non-EU/EFTA countries should be allowed to work in Switzerland (increase third-country quota)."
# VARIANT_TEXT = "More qualified workers from non-EU/EFTA countries should not be allowed to work in Switzerland (increase third-country quota)."

# BASE_TEXT = "Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. "
# VARIANT_TEXT = "Foreign nationals who have lived in Switzerland for at least ten years should not be granted the right to vote and stand for election at the municipal level."

# BASE_TEXT = "Cannabis use should be legalized. "
# VARIANT_TEXT = "Cannabis use should not be legalized."

# BASE_TEXT = "Doctors should be allowed to administer direct active euthanasia. "
# VARIANT_TEXT = "Doctors should not be allowed to administer direct active euthanasia."

# BASE_TEXT = "Same-sex couples should have the same rights as heterosexual couples in all areas. "
# VARIANT_TEXT = "Same-sex couples should not have the same rights as heterosexual couples in all areas."

# BASE_TEXT = "There should be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "
# VARIANT_TEXT = "There should not be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "

# BASE_TEXT = "Private households should be free to choose their electricity supplier (complete liberalization of the electricity market). "
# VARIANT_TEXT = "Private households should not be free to choose their electricity supplier (complete liberalization of the electricity market)."

# BASE_TEXT = "There should be stricter controls on equal pay for women and men. "
# VARIANT_TEXT = "There should not be stricter controls on equal pay for women and men."

# BASE_TEXT = "The protection regulations for large predators (lynx, wolf, bear) should be relaxed. "
# VARIANT_TEXT = "The protection regulations for large predators (lynx, wolf, bear) should not be relaxed. "

# BASE_TEXT = "There should be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas). "
# VARIANT_TEXT = "There should not be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas)."

# BASE_TEXT = "Young people over the age of 16 are to be allowed to vote in Bundestag elections."
# VARIANT_TEXT = "Young people aged 16 and over should not be allowed to vote in federal elections."

# BASE_TEXT = "The right of recognized refugees to join their families is to be abolished."
# VARIANT_TEXT = "The right of recognized refugees to family reunification should not be abolished."

# BASE_TEXT = "Donations from companies to political parties should continue to be permitted."
# VARIANT_TEXT = "Donations from companies to political parties should not be permitted."

# BASE_TEXT = "Federal authorities are to take linguistic account of different gender identities in their publications."
# VARIANT_TEXT = "Federal authorities should not use different gender identities in their publications."

# BASE_TEXT = "Female civil servants are to be allowed to wear headscarves while on duty."
# VARIANT_TEXT = "Female civil servants should not be allowed to wear headscarves on duty."

# BASE_TEXT = "The federal government is to provide more financial support for projects to combat anti-Semitism."
# VARIANT_TEXT = "The federal government should not provide more financial support for projects to combat anti-Semitism."

# BASE_TEXT = "The controlled sale of cannabis is to be generally permitted."
# VARIANT_TEXT = "The controlled sale of cannabis should not be permitted."

# BASE_TEXT = "Organic agriculture should be promoted more strongly than conventional agriculture."
# VARIANT_TEXT = "Organic farming should not be promoted more than conventional farming."

# BASE_TEXT = "Islamic associations are to be able to be recognized by the state as religious communities."
# VARIANT_TEXT = "Islamic associations should not be able to be recognized by the state as religious communities."

# BASE_TEXT = "The state should take measures to redistribute wealth from the rich to the poor."
# VARIANT_TEXT = "The state should not take measures to redistribute wealth from the rich to the poor."

# BASE_TEXT = "The government must increase spending on public health care, even if this means increasing taxes."
# VARIANT_TEXT = "The government should not increase spending on the public health system even if this means increasing taxes."

# BASE_TEXT = "Taxes on fossil fuels must be raised to finance the Green Transition."
# VARIANT_TEXT = "Taxes on fossil fuels should not be raised to finance the Ecological Transition."

# BASE_TEXT = "To better defend Spain's interests in Europe we must recover more sovereignty."
# VARIANT_TEXT = "In order to better defend Spain's interests in Europe, we should not recover more sovereignty."

# BASE_TEXT = "The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum."
# VARIANT_TEXT = "The best way to solve the conflict in Catalonia is that its citizens cannot vote on their future in a referendum."

# BASE_TEXT = "Negotiating with pro-independence supporters weakens the State."
# VARIANT_TEXT = "Negotiating with the independentistas does not weaken the State."

# BASE_TEXT = "The right to self-determination must be recognized by the Constitution."
# VARIANT_TEXT = "The right of self-determination should not be recognized by the Constitution."

# BASE_TEXT = "Spain should be more tolerant with illegal migration."
# VARIANT_TEXT = "Spain should not be more tolerant of illegal immigration."

# BASE_TEXT = "Housing prices must be regulated to ensure access for all people."
# VARIANT_TEXT = "Housing prices should not be regulated to guarantee access to all people."

# BASE_TEXT = "Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities)."
# VARIANT_TEXT = "Gender identity should not be influenced by environmental influences (e.g. media content, sensitising activities)."

# BASE_TEXT = "Stronger state regulation of the work of NGOs supported by foreign organisations is needed."
# VARIANT_TEXT = "There is no need for stronger state regulation of the work of NGOs supported by foreign organisations."

# BASE_TEXT = "Hungary should join the European Public Prosecutor's Office."
# VARIANT_TEXT = "Hungary should not join the European Public Prosecutor's Office."

# BASE_TEXT = "Stricter regulation of interception software (e.g. Pegasus) is needed (e.g. subject to judicial authorisation)."
# VARIANT_TEXT = "There is no need for stricter regulation of interception software (e.g. Pegasus) (e.g. subject to judicial authorisation)."

# BASE_TEXT = "The age of compulsory schooling should be raised back to 18."
# VARIANT_TEXT = "The age of compulsory education should not be raised back to 18."

# BASE_TEXT = "The Hungarian government should ratify the Istanbul Convention, which combats violence against women and domestic violence."
# VARIANT_TEXT = "The Hungarian government should not ratify the Istanbul Convention against violence against women and domestic violence."

# BASE_TEXT = "One effective way to reduce rents is to conclude favourable gas supply contracts with Russia."
# VARIANT_TEXT = "The conclusion of favourable gas supply contracts with Russia is not an effective way of reducing rationing."

# BASE_TEXT = "Comprehensive public procurement reform is needed (e.g. opening up large-scale centralised public procurement to smaller firms)."
# VARIANT_TEXT = "There is no need for comprehensive public procurement reform (e.g. opening up large-scale centralised public procurement to smaller firms)."

# BASE_TEXT = "Public employment helps people re-enter the labour market."
# VARIANT_TEXT = "Public works do not help people to re-enter the labour market."

# BASE_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation."
# VARIANT_TEXT = "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is not the right way to fight inflation."

# BASE_TEXT = "State regulation of the rental housing market is not necessary."
# VARIANT_TEXT = "Public regulation of the rental housing market is needed."

# BASE_TEXT = "The use of medical cannabis should be legalised in Hungary."
# VARIANT_TEXT = "Medical cannabis should not be legalised in Hungary."

# BASE_TEXT = "The President of the Hungarian Republic should be directly elected."
# VARIANT_TEXT = "The President of the Hungarian Republic should not be directly elected."

# BASE_TEXT = "Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner-take-all compensation, extension of postal voting) is needed."
# VARIANT_TEXT = "There is no need for a comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner compensation, extension of postal voting)."

# BASE_TEXT = "A legal framework for primary elections should be provided."
# VARIANT_TEXT = "There is no need to provide a legal framework for primary elections."

# BASE_TEXT = "Internet access should be free for all."
# VARIANT_TEXT = "Internet should not be free for all."

# BASE_TEXT = "Polluting companies should be taxed more heavily."
# VARIANT_TEXT = "Polluting companies should not be subject to higher taxes."

# BASE_TEXT = "In larger cities, car traffic should be limited through various measures (P+R parking, construction of cycle paths, improvement of public transport)."
# VARIANT_TEXT = "In larger cities, there is no need to restrict car traffic through various measures (P+R parking, building cycle paths, improving public transport)."

# BASE_TEXT = "The redevelopment of urban green spaces (e.g. the Liget project in Budapest) needs a broad social dialogue."
# VARIANT_TEXT = "The redevelopment of urban green areas (e.g. the Liget project in Budapest) does not require a broad social dialogue."

# BASE_TEXT = "An independent ministry for the environment is needed."
# VARIANT_TEXT = "There is no need for a separate environment ministry."

# BASE_TEXT = "An animal rights commissioner should be introduced."
# VARIANT_TEXT = "No need to introduce an animal rights commissioner."

# BASE_TEXT = "The European Union should have a common foreign policy."
# VARIANT_TEXT = "The European Union should not have a common foreign policy."

# BASE_TEXT = "European economic integration has gone too far: member states should regain more autonomy."
# VARIANT_TEXT = "European economic integration has not gone too far: member states should not regain more autonomy."

# BASE_TEXT = "Children, born in Italy to foreign citizens and who have completed schooling should be granted Italian citizenship (ius scholae)."
# VARIANT_TEXT = "Children, born in Italy to foreign nationals and who have completed school, should not be granted Italian citizenship (ius scholae)."

# BASE_TEXT = "More civil rights should be granted to homosexual, bisexual, transgender (LGBT+) people."
# VARIANT_TEXT = "Homosexual, bisexual, transgender (LGBT+) people should not be granted more civil rights."

# BASE_TEXT = "Citizens should be guaranteed freedom of choice in end-of-life matters (euthanasia)."
# VARIANT_TEXT = "Citizens should not be guaranteed freedom of choice in end-of-life matters (euthanasia)."

# BASE_TEXT = "Recreational use of marijuana/cannabis should be allowed."
# VARIANT_TEXT = "Recreational use of marijuana/cannabis should not be allowed."

# BASE_TEXT = "Health care should be managed only by the state and not by private individuals."
# VARIANT_TEXT = "Health care should not only be managed by the state, but also by private individuals."

# BASE_TEXT = "The construction of Major Works is a priority for Italy."
# VARIANT_TEXT = "The construction of Major Works is not a priority for Italy."

# BASE_TEXT = "Regasifiers are necessary infrastructure for Italy."
# VARIANT_TEXT = "Regasifiers are not necessary infrastructure for Italy."

# BASE_TEXT = "Italy should keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."
# VARIANT_TEXT = "Italy should not keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."

# BASE_TEXT = "Organizers of events should be able to request a vaccination certificate upon entry."
# VARIANT_TEXT = "Event organizers should not require vaccination certificates at entry."

# BASE_TEXT = "The Netherlands should exit the European Union (EU)."
# VARIANT_TEXT = "The Netherlands should not leave the European Union (EU)."

# BASE_TEXT = "The Netherlands should build a new nuclear power plant."
# VARIANT_TEXT = "The Netherlands should not build a new nuclear power plant."

# BASE_TEXT = "Households with two partners, one of whom works, should receive the same tax benefits as households with two working partners."
# VARIANT_TEXT = "Households with two partners of which one works should not receive the same tax benefit as households with two working partners."

# BASE_TEXT = "The Dutch government should apologize for the historical slave trade."
# VARIANT_TEXT = "The Dutch government should not apologize for the slave trade in the past."

# BASE_TEXT = "Citizens should have the opportunity to block laws passed by parliament through a referendum."
# VARIANT_TEXT = "Citizens should not be allowed to stop laws passed by parliament through a referendum."

# BASE_TEXT = "There should be fewer options for community service sentences instead of prison sentences."
# VARIANT_TEXT = "There should not be fewer opportunities to impose community service instead of prison sentences."

# BASE_TEXT = "The government should make Dutch-language education more frequently mandatory at universities and colleges."
# VARIANT_TEXT = "The government should no longer make education in Dutch compulsory at universities and colleges."

# BASE_TEXT = "A middle school should be established so that students make a choice between vocational education, general secondary education, or pre-university education at a later age."
# VARIANT_TEXT = "There should be no middle school, so that pupils do not have to choose between vmbo, havo or vwo at a later age."

# BASE_TEXT = "Social transfers should be increased to reduce the effects of inflation on citizens."
# VARIANT_TEXT = "Social transfers should not be increased to limit the effects of inflation on citizens."

# BASE_TEXT = "The state should finance private visits to specialists if the waiting time at a public facility exceeds three months."
# VARIANT_TEXT = "The state should not finance private visits to specialists if the waiting time at a public facility exceeds three months."

# BASE_TEXT = "The state should provide a free nursery place for every child."
# VARIANT_TEXT = "The state should not provide a free nursery place for every child."

# BASE_TEXT = "The state should build low-rent apartments for rent."
# VARIANT_TEXT = "The state should not build low-income rental housing."

# BASE_TEXT = "The independence of the judiciary from parliament and the government should be strengthened."
# VARIANT_TEXT = "The independence of the judiciary from parliament and government should not be strengthened."

# BASE_TEXT = "The powers of local governments should be increased at the expense of the central government."
# VARIANT_TEXT = "The powers of local governments should not be increased at the expense of the central government."

# BASE_TEXT = "Poland should move away from coal mining no later than 2040."
# VARIANT_TEXT = "Poland should not move away from coal mining by 2040."

# BASE_TEXT = "Poland should have grain imports from Ukraine blocked."
# VARIANT_TEXT = "Poland should not lead to the blocking of grain imports from Ukraine."

BASE_TEXT = "The powers of the secret services to track the activities of citizens on the Internet should be limited."
VARIANT_TEXT = "The powers of the secret services to track the activities of citizens on the Internet should not be restricted."



print_top_layers = 36
TEMP_FOR_PROBS = 1.0
EPS = 1e-9
QWEN_ENABLE_THINKING = False


def flip_probs_1_to_7(p: torch.Tensor) -> torch.Tensor:
    idx = torch.tensor([6, 5, 4, 3, 2, 1, 0], device=p.device)
    return p.index_select(dim=-1, index=idx)


def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model) -> List[Tuple[int, str, nn.Module]]:
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen3DecoderLayer):
            layers.append((len(layers), name, mod))
    if not layers:
        raise RuntimeError(
            "No Qwen3DecoderLayer found via named_modules(). "
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


def _apply_qwen_chat_template_or_fallback(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> str:
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return text

    return system_prompt.strip() + "\n\n" + user_prompt


def encode_for_next_token(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> EncodedChat:
    text = _apply_qwen_chat_template_or_fallback(
        tokenizer, system_prompt, user_prompt, enable_thinking=enable_thinking
    )
    enc = tokenizer(text, return_tensors="pt")
    dev = get_input_device(model)

    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)

    seq_len = input_ids.shape[-1]
    answer_pos = seq_len - 1

    digit_ids: List[int] = []
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
    logits = out.logits[:, enc.answer_pos, :].squeeze(0)
    return logits


def digit_logit_slice(logits: torch.Tensor, digit_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(digit_ids, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def pick_target_digit_id(logits_clean_digits: torch.Tensor, digit_ids: List[int]) -> int:
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


def collect_clean_cache(model, enc_clean: EncodedChat) -> CleanCache:
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
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Qwen3MLP):
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
            if isinstance(sub, Qwen3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Qwen3MLP):
                hooks.append(sub.register_forward_hook(mlp_patch_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def w_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)


def normalized_restoration(dist_fn, p_clean, p_corrupt, p_patched, eps=1e-12):
    p_target = flip_probs_1_to_7(p_clean)
    d0 = dist_fn(p_target, p_corrupt)
    dp = dist_fn(p_target, p_patched)
    R = 1.0 - dp / (d0 + eps)
    return torch.where(d0 <= eps, torch.full_like(R, float("nan")), R)


def run_activation_patching(base_text: str, variant_text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    enc_clean = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(base_text), enable_thinking=QWEN_ENABLE_THINKING
    )
    enc_corrupt = encode_for_next_token(
        tokenizer, model, SYSTEM_PROMPT, build_user_prompt(variant_text), enable_thinking=QWEN_ENABLE_THINKING
    )

    with torch.no_grad():
        logits_clean = forward_logits_only(model, enc_clean)
        logits_corrupt = forward_logits_only(model, enc_corrupt)
        logits_clean_digits = digit_logit_slice(logits_clean, enc_clean.digit_ids)
        logits_corrupt_digits = digit_logit_slice(logits_corrupt, enc_corrupt.digit_ids)

    target_digit_id = pick_target_digit_id(logits_clean_digits, enc_clean.digit_ids)
    clean_target_logit = logits_clean[target_digit_id].item()
    corrupt_target_logit = logits_corrupt[target_digit_id].item()

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    print(f"[Target digit id] {target_digit_id}  ({tokenizer.decode([target_digit_id])})  (for reference)")
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
                patched_probs = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)

            _ = objective_from_logits_full(logits_patched, enc_corrupt, clean_probs, TEMP_FOR_PROBS).item()
            r = normalized_restoration(w_1d, clean_probs, corrupt_probs, patched_probs)
            results.append((l, float(r)))
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
    print_top("[Patch - MLP  - top layers]", mlp_results)

    return True


def set_global_determinism(seed: int = 42, single_thread: bool = True):
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


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    set_global_determinism(0, single_thread=True)
    _ = run_activation_patching(BASE_TEXT, VARIANT_TEXT)
