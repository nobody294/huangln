import os
import math
import random
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3Attention,
    Gemma3MLP
)

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are a voter being asked for opinions. "
    "Your only job is to rate policy statements on a 1-7 Likert scale."
)

# Put all flip pairs of ONE wording rule here (for aggregation)
# PAIRS: List[Tuple[str, str]] = [
#     ("Childcare should be free for all parents for at least three days a week.",
#      "Child care should remain paying for all parents at least three days a week."),

#     ("There should be an additional tax on purchasing meat.",
#      "There should be less tax on buying meat."),

#     ("Instead of the existing health insurance companies, there should be a national healthcare fund for everyone.",
#      "A national health care fund for everyone should not replace existing health insurance companies."),

#     ("The national government, rather than provinces and municipalities, should decide where new residential areas are built.",
#      "Instead of the national government, provinces and municipalities should continue to decide where new housing developments are built."),

#     ("Houses should be built on land currently used for agriculture.",
#      "Land now used for agriculture should only be used for agriculture."),

#     ("There should be fewer options for community service sentences instead of prison sentences.",
#      "There should be more opportunities to impose community service instead of prison sentences."),

#     ("The Netherlands should introduce an additional flight tax for short-distance flights.",
#      "The Netherlands should introduce fewer flight taxes for short-haul flights."),

#     ("The government should make Dutch-language education more frequently mandatory at universities and colleges.",
#      "The government should make education in Dutch less compulsory at universities and colleges."),

#     ("People who consider their lives complete should be able to receive assistance with suicide.",
#      "People who find their lives complete should be denied assisted suicide."),

#     ("New residential areas should consist of at least 40 percent green space.",
#      "New housing developments should consist of up to 40 percent social housing."),

#     ("The growth of Islam is a threat to Spain's security.",
#      "The growth of Islam is harmless to Spain's security."),

#     ("Current gender policies are biased against men.",
#      "Current gender policies benefit men."),

#     ("The efficiency of public services improves when they are privatized.",
#      "Efficiency in the provision of public services worsens when they are privatized."),

#     ("A permanent tax on large fortunes and assets is necessary.",
#      "It is necessary to eliminate the permanent tax on large fortunes and wealth."),

#     ("Climate change is solely attributable to human action.",
#      "Climate change is only attributable to natural action."),

#     ("The future Spanish government should increase irrigated agricultural areas by means of large water transfers.",
#      "The future Spanish government should reduce agricultural irrigated areas through large water transfers."),

#     ("Negotiating with pro-independence supporters weakens the State.",
#      "Negotiating with the independentistas strengthens the State."),

#     ("Covid-19 vaccines are to continue to be protected by patents.",
#      "Vaccines against Covid-19 should not be patented and should be freely available."),

#     ("All employed persons are to be required to be insured in the statutory pension scheme.",
#      "All workers should be optionally insured in the statutory pension scheme."),

#     ("The traditional family of father, mother and children is to be promoted more strongly than other living arrangements.",
#      "Other cohabiting couples are to be promoted more than the traditional family of father, mother and children."),

#     ("Students should receive BAföG regardless of their parents' income.",
#      "Students should receive BAföG, depending on their parents' income."),

#     ("The Nord Stream 2 Baltic Sea pipeline, which transports gas from Russia to Germany, is to be allowed to go into operation as planned.",
#      "The \"\"Nord Stream 2\"\" Baltic Sea pipeline, which transports gas from Russia to Germany, should not be allowed to go into operation."),

#     ("The registration of new cars with combustion engines should also be possible in the long term.",
#      "The registration of new cars with combustion engines should be banned in the long term."),

#     ("The federal government is to be given more responsibilities in school policy.",
#      "The federal government should retain fewer responsibilities in school policy."),

#     ("The state should continue to collect church tax for religious communities.",
#      "The state should stop collecting church tax for religious communities."),

#     ("Facial recognition software should be allowed to be used for video surveillance in public places.",
#      "The use of facial recognition software for video surveillance in public places should be prohibited."),

#     ("The government-set price for CO2 emissions from heating and driving is to rise more than planned.",
#      "The price set by the state for CO2 emissions from heating and driving is to increase less than planned."),

#     ("The result of any nationwide referendum should be binding regardless of turnout.",
#      "The result of any nationwide referendum should be binding depending on turnout."),

#     ("Poland should adopt the migrant relocation solutions adopted by the European Union.",
#      "Poland should reject the migrant relocation solutions adopted by the European Union."),

#     ("The EU's rule of law mechanism threatens Hungary's sovereignty.",
#      "The EU's rule of law mechanism protects Hungary's sovereignty."),

#     ("Hungary should decide by referendum whether to remain part of the EU.",
#      "Hungary should decide to remain part of the EU without consulting the electorate."),

#     ("Stronger state regulation of the work of NGOs supported by foreign organisations is needed.",
#      "Weaker state regulation of the work of foreign-funded NGOs is needed."),

#     ("The age of compulsory schooling should be raised back to 18.",
#      "The age of compulsory education should remain 16."),

#     ("Political influence has been reduced by changing the university model (reorganisation into a trust).",
#      "Political influence has increased with the change in the university model (reorganisation into a trust)."),

#     ("One effective way to reduce rents is to conclude favourable gas supply contracts with Russia.",
#      "The conclusion of favourable gas supply contracts with Russia is an inappropriate way of reducing overheads."),

#     ("Increase the contribution of the wealthier to the public purse (abolition of the one-band tax).",
#      "The more wealthy should contribute less to the public burden (abolition of the one-band tax)."),

#     ("A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation.",
#      "A price freeze on some basic foodstuffs (e.g. chicken tails, milk) is ineffective in combating inflation."),

#     ("The President of the Hungarian Republic should be directly elected.",
#      "The President of the Hungarian Republic should be elected by the Parliament."),

#     ("A legal framework for primary elections should be provided.",
#      "A civilian regulatory framework for primaries should be provided."),

#     ("Voting age for elections should be 16.",
#      "Voting age for elections should be 18."),

#     ("A heritage tax one's wealth should be introduced.",
#      "A wealth tax on great wealth should be cut."),

#     ("Health care should be managed only by the state and not by private individuals.",
#      "Health care should be managed only by private individuals and not by the state."),

#     ("The citizenship allowance is a measure that should be cancelled.",
#      "The citizenship income is a measure that should be continued."),

#     ("Beach concessions to private individuals should be time-limited.",
#      "Beach concessions to private individuals should be of continuous duration over time."),

#     ("Italy should build more incinerators/thermal power plants.",
#      "Italy should stop building incinerators/thermal power plants."),

#     ("Drilling is necessary to find more energy resources.",
#      "Drilling is irrelevant to finding more energy resources."),

#     ("Sanctions against Russia should be tougher.",
#      "Sanctions against Russia should be eased."),

#     ("Taxes collected at the regional level should be fully retained in the region itself.",
#      "Taxes collected at the regional level should be fully redistributed among the regions."),

#     ("The Federal Council's ability to restrict private and economic life in the event of a pandemic should be more limited.",
#      "The Federal Council's ability to restrict private and economic life in the event of a pandemic should be expanded."),

#     ("The federal government should raise the requirements for the high school.",
#      "The federal government should decrease the requirements for the high school."),

#     ("A third official gender should be introduced alongside \"\"female\"\" and \"\"male\"\".",
#      "A third official gender should be disregarded alongside \"\"female\"\" and \"\"male\"\"."),

#     ("Same-sex couples should have the same rights as heterosexual couples in all areas.",
#      "Same-sex couples should have more rights as heterosexual couples in all areas."),

#     ("The differences between cantons with high and low financial capacity should be further reduced through fiscal equalization.",
#      "The differences between cantons with high and low financial capacity should be further increased through fiscal equalization."),

#     ("Direct payments should only be granted to farmers with proof of ecological performance.",
#      "Direct payments should be granted to all farmers without requiring proof of ecological performance."),

#     ("The army's target number of soldiers should expand to at least 120,000.",
#      "The army's target number of soldiers should be decreased."),

#     ("The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine).",
#      "The Federal Council should be forbidden to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine)."),

#     ("Automatic facial recognition should be banned in public spaces.",
#      "Automatic facial recognition should be allowed in public spaces."),

#     ("Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border.",
#      "Switzerland should keep the Schengen agreement with the EU. There's no need for more security checks directly on the border."),
# ]

PAIRS: List[Tuple[str, str]] = [
    ("There should be a ban on single-use plastic and non-recyclable plastics. ",
     "There should be an incentive to use single-use plastic and non-recyclable plastics."),

    ("There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). ",
     "The government should ignore measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "),

    ("There should be closer relations with the European Union (EU). ",
     "There should be more limited relations with the European Union (EU)."),

    ("Switzerland should strive for a comprehensive free trade agreement (including agriculture) with the USA. ",
     "Switzerland should neglect a comprehensive free trade agreement (including agriculture) with the USA."),

    ("Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. ",
     "Companies should be ignore whether their subsidiaries and suppliers operating abroad comply with social and environmental standards."),

    ("Switzerland should terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons. ",
     "Switzerland should mantain the Bilateral Agreements with the EU for free movement of persons."),

    ("Switzerland should return to a strict interpretation of neutrality (renounce economic sanctions to a large extent). ",
     "Switzerland should maintain a laxer interpretation of neutrality (renounce economic sanctions to a large extent)."),

    ("There should be an increase in the retirement age. ",
     "There should be a decrease in the retirement age."),

    ("The federal government should allocate more funding for health insurance premium subsidies. ",
     "The federal government should revoke funding for health insurance premium subsidies."),

    ("The Swiss mobile network should be equipped throughout the country with the latest technology (currently 5G standard). ",
     "The Swiss mobile network should be kept the same throughout the country with the latest technology (currently 5G standard)."),

    ("For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should be eliminated.",
     "For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should be kept."),

    ("As part of the reform of the BVG (occupational pension plan), pensions are to be reduced (lowering the minimum conversion rate from 6.8% to 6%). ",
     "As part of the reform of the BVG (occupational pension plan), pensions are to be increased (lowering the minimum conversion rate from 6.8% to 6%). "),

    ("Paid parental leave should be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave. ",
     "Paid parental leave should be reduced under today's 14 weeks of maternity leave and two weeks of paternity leave."),

    ("The federal government should provide more financial support for public housing construction. ",
     "The federal government should provide less financial support for public housing construction."),

    ("Compulsory vaccination of children should be introduced based on the Swiss vaccination plan. ",
     "Vaccination of children should be voluntary and not mandated by the Swiss vaccination plan."),

    ("There should be an introduction of a tax on foods containing sugar (sugar tax). ",
     "There should be a disregard on a tax on foods containing sugar (sugar tax)."),

    ("Insured persons should contribute more to health care costs (e.g., increase the minimum deductible).",
     "Insured persons should contribute less to health care costs (e.g., decrease the minimum deductible)."),

    ("The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services). ",
     "The federal government should be removed the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."),

    ("According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in regular classes. ",
     "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in separate classes. "),

    ("The federal government should be given additional powers in the area of digitization of government services in order to be able to impose binding directives and standards on the cantons.",
     "The federal government should be given less power in the area of digitization of government services in order to be able to impose binding directives and standards on the cantons."),

    ("The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). ",
     "The state should neglect equal educational opportunities (e.g., regarding subsidized remedial courses for students from low-income families). "),

    ("The conditions for naturalization should be relaxed (e.g., shorter residence period). ",
     "The conditions for naturalization should be tightened (e.g., longer residence period). "),

    ("More qualified workers from non-EU/EFTA countries should be allowed to work in Switzerland (increase third-country quota).",
     "More qualified workers from non-EU/EFTA countries should be forbidden to work in Switzerland (decrease third-country quota)."),

    ("Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. ",
     "Foreign nationals who have lived in Switzerland for at least ten years should be refused the right to vote and stand for election at the municipal level."),

    ("Cannabis use should be legalized. ",
     "Cannabis use should be kept ilegal."),

    ("Doctors should be allowed to administer direct active euthanasia. ",
     "Doctors should be fobidden to administer direct active euthanasia."),

    ("There should be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). ",
     "There should be a laxer regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). "),

    ("There should be tax cuts at the federal level over the next four years. ",
     "There should be tax increase at the federal level over the next four years."),

    ("A minimum wage of CHF 4,000 for all full-time employees should be introduced. ",
     "A minimum wage of CHF 4,000 for all full-time employees should be disregarded. "),

    ("There should be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). ",
     "There should be laxer regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "),

    ("Private households should be free to choose their electricity supplier (complete liberalization of the electricity market). ",
     "Private households should be limited in choosing their electricity supplier (full regulation of the electricity market)."),

    ("There should be stricter controls on equal pay for women and men. ",
     "There should be laxer controls on equal pay for women and men."),

    ("Busy sections of highways should be widened. ",
     "Busy sections of highways should be narrowed."),

    ("There should be a popular initiative aims to reduce television and radio fees (CHF 200 per household, exemption for businesses).",
     "There should be a popular initiative aims to increase television and radio fees (CHF 200 per household, exemption for businesses)."),

    ("Switzerland should ban the registration of new passenger cars with internal combustion engines starting in 2035. ",
     "Switzerland should relax the registration of new passenger cars with internal combustion engines starting in 2035."),

    ("To achieve climate targets, incentives and target agreements should be relied on exclusively, rather than bans and restrictions. ",
     "To achieve climate targets, incentives and target agreements should be relied on bans and restrictions. "),

    ("The construction of new nuclear power plants should be allowed again. ",
     "The construction of new nuclear power plants should be denied again."),

    ("The state should guarantee a comprehensive public service offering also in rural regions. ",
     "The state should ignore a comprehensive public service offering also in rural regions."),

    ("Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be introduced. ",
     "Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be disregarded."),

    ("The protection regulations for large predators (lynx, wolf, bear) should be relaxed. ",
     "The protection regulations for large predators (lynx, wolf, bear) should be made stricter. "),

    ("There should be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas). ",
     "There should be laxer animal welfare regulations for livestock (e.g. only temporary access to outdoor areas)."),

    ("30% of Switzerland's land area should be dedicated to preserving biodiversity?. ",
     "Switzerland should ignore the allocation of any specific percentage of its land area to preserving biodiversity."),

    ("The voting age should be lowered to 16. ",
     "The voting age should be increased."),

    ("It should be possible to hold a referendum on federal spending above a certain amount (optional financial referendum). ",
     "It should be kept impossible to hold a referendum on federal spending above a certain amount (optional financial referendum)."),

    ("The Swiss Armed Forces should expand their cooperation with NATO. ",
     "The Swiss Armed Forces should reduce their cooperation with NATO."),

    ("A general speed limit is to apply on all highways.",
     "Unlimited speed should be allowed on all highways."),

    ("Donations from companies to political parties should continue to be permitted.",
     "Donations from companies to political parties should be prohibited."),

    ("Federal authorities are to take linguistic account of different gender identities in their publications.",
     "Federal authorities should not use gender-neutral language in their publications."),

    ("Female civil servants are to be allowed to wear headscarves while on duty.",
     "Female civil servants should generally be banned from wearing headscarves on duty."),

    ("Germany is to increase its defense spending.",
     "Germany should reduce its defense spending."),

    ("The federal government is to provide more financial support for projects to combat anti-Semitism.",
     "The federal government should provide less financial support for projects to combat anti-Semitism."),

    ("Chinese companies should not be allowed to receive contracts for the expansion of the communications infrastructure in Germany.",
     "Chinese companies are to be allowed to receive contracts for the expansion of the communications infrastructure in Germany."),

    ("The controlled sale of cannabis is to be generally permitted.",
     "The controlled sale of cannabis should be prohibited."),

    ("Germany is to leave the European Union.",
     "Germany should remain a member of the European Union."),

    ("The state lists of the parties for the elections to the German Bundestag are to have to be filled alternately by women and men.",
     "The state lists of the parties for the elections to the German Bundestag should not take into account an equal distribution of men and women."),

    ("A tax is to be levied again on high assets.",
     "The levying of a tax on high assets should be discontinued."),

    ("Young people over the age of 16 are to be allowed to vote in Bundestag elections.",
     "Only adults (18 and over) should be allowed to vote in federal elections."),

    ("Married couples without children should continue to receive tax breaks.",
     "Only married couples with children should receive preferential tax treatment."),

    ("Organic agriculture should be promoted more strongly than conventional agriculture.",
     "Conventional agriculture should be promoted more than organic farming."),

    ("Islamic associations are to be able to be recognized by the state as religious communities.",
     "Islamic associations should be rejected by the state as religious communities."),

    ("The debt brake in the Basic Law is to be retained.",
     "The debt brake in the Basic Law is to be lifted."),

    ("Asylum is to continue to be granted only to politically persecuted persons.",
     "Asylum is to be granted to more people than just those who are politically persecuted."),

    ("The statutory minimum wage is to be increased to at least 12 euros by 2022 at the latest.",
     "The statutory minimum wage is to be reduced by 2022 at the latest."),

    ("Air traffic is to be taxed more heavily.",
     "Air traffic is to be taxed at a lower rate."),

    ("Subsidies for wind energy are to be ended.",
     "The promotion of wind energy should be expanded."),

    ("The ability of landlords to increase housing rents is to be more strictly limited by law.",
     "Landlords should be allowed to increase rents without legal restrictions."),

    ("The phase-out of coal-fired power generation planned for 2038 is to be brought forward.",
     "The phase-out of coal-fired power generation planned for 2038 should be reversed."),

    ("The right of recognized refugees to join their families is to be abolished.",
     "The right of recognized refugees to family reunification is to be extended."),

    ("Limiting rights and freedoms is necessary to combat organized crime.",
     "It is necessary to expand rights and freedoms to combat organized crime."),

    ("Governments should intervene as little as possible in the economy.",
     "Governments should intervene as much as possible in the economy."),

    ("Taxes on fossil fuels must be raised to finance the Green Transition.",
     "Taxes on fossil fuels should be reduced and the Ecological Transition should be ignored."),

    ("To better defend Spain's interests in Europe we must recover more sovereignty.",
     "To better defend Spain's interests in Europe, we must cede sovereignty."),

    ("Spanish government should promote the strengthening of NATO in Europe.",
     "The Spanish government should promote the weakening of NATO in Europe."),

    ("The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum.",
     "The worst way to solve the conflict in Catalonia is to allow its citizens to vote on their future in a referendum."),

    ("Spain's territorial decentralization must be deepened.",
     "The centralization of power in Spain must be deepened."),

    ("The right to self-determination must be recognized by the Constitution.",
     "The right of self-determination must be ignored by the Constitution."),

    ("The policies of linguistic immersion in the native language of bilingual Autonomous Communities endanger Spanish.",
     "The policies of linguistic immersion in the own language of the bilingual Autonomous Communities protect Spanish."),

    ("It is necessary to repeal the Law of Democratic Memory passed during this legislature.",
     "It is necessary to pass the Law of Democratic Memory during this legislature."),

    ("Spain should be more tolerant with illegal migration.",
     "Spain should be more intolerant of illegal immigration."),

    ("Immigrants should pay for their own health services.",
     "Immigrants should receive free health services."),

    ("Housing prices must be regulated to ensure access for all people.",
     "Housing prices should be left to the free market."),

    ("It should be easier for companies to fire workers.",
     "It should be harder for companies to lay off workers."),

    ("The state should take measures to redistribute wealth from the rich to the poor.",
     "The state must take measures to increase the gap between rich and poor."),

    ("The government must increase spending on public health care, even if this means increasing taxes.",
     "The government should decrease spending on public health care so as not to increase taxes."),

    ("The working day should be reduced without reducing workers' wages.",
     "Working hours should be increased without increasing workers' salaries."),

    ("Education spending should be increased to at least the OECD average of 5.2 per cent (GDP).",
     "Spending on education is sufficient."),

    ("Teachers' salaries should be doubled.",
     "Teachers' salaries are sufficient."),

    ("Only men and women should be allowed to marry.",
     "Same-sex couples should be allowed to marry."),

    ("Parties should strive for a closer ratio of men to women when drawing up lists.",
     "Political parties prevent the proportion of men and women on the lists from converging."),

    ("The state should take targeted measures to promote equal participation of fathers and mothers in child-rearing.",
     "The state takes targeted measures to prevent fathers and mothers from sharing equally in child-rearing."),

    ("The Hungarian government should ratify the Istanbul Convention, which combats violence against women and domestic violence.",
     "The Hungarian government should reject the ratification of the Istanbul Convention, which combats violence against women and domestic violence."),

    ("Comprehensive public procurement reform is needed (e.g. opening up large-scale centralised public procurement to smaller firms).",
     "Comprehensive public procurement reform is unnecessary (e.g. opening up large-scale centralised public procurement to smaller companies)."),

    ("Hungarian foreign policy should be guided solely by Hungarian economic interests.",
     "Hungarian foreign policy should also take into account foreign economic interests."),

    ("Public employment helps people re-enter the labour market.",
     "Public employment is a barrier to re-entering the labour market."),

    ("State regulation of the rental housing market is not necessary.",
     "Public regulation of the rental housing market is required."),

    ("The current three-month unemployment benefit should be extended.",
     "Cutting the current three-month unemployment benefit is unnecessary."),

    ("A family tax credit is a better way to support families than increasing the family allowance.",
     "The family tax allowance is a worse way of supporting families than increasing the family allowance."),

    ("The use of medical cannabis should be legalised in Hungary.",
     "Make the use of medical cannabis illegal in Hungary."),

    ("An independent Ministry of Health should be established.",
     "The creation of an independent Ministry of Health is unnecessary"),

    ("Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner-take-all compensation, extension of postal voting) is needed.",
     "Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner's compensation, extension of postal voting) is unnecessary."),

    ("Internet access should be free for all.",
     "Internet should be free for all."),

    ("Polluting companies should be taxed more heavily.",
     "Higher taxes on polluting companies should be avoided."),

    ("In larger cities, car traffic should be limited through various measures (P+R parking, construction of cycle paths, improvement of public transport).",
     "In larger cities, it is unnecessary to restrict car traffic by various measures (P+R parking, building cycle paths, improving public transport)."),

    ("The redevelopment of urban green spaces (e.g. the Liget project in Budapest) needs a broad social dialogue.",
     "In the case of the redevelopment of urban green areas (e.g. the Liget project in Budapest), a broad social dialogue is unjustified."),

    ("An independent ministry for the environment is needed.",
     "A separate environment ministry is unnecessary."),

    ("Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities).",
     "Gender identity is formed independently of environmental influences (e.g. media content, sensitising activities)."),

    ("An animal rights commissioner should be introduced.",
     "The introduction of an animal rights commissioner is unnecessary."),

    ("Hungary should join the European Public Prosecutor's Office.",
     "Hungary should withdraw from the European Public Prosecutor's Office."),

    ("Disclosure of the origin of criminals is needed for more effective law enforcement.",
     "Disclosing the origin of criminals does not help more effective law enforcement."),

    ("European integration is all in all a positive process.",
     "European integration is an all-negative process."),

    ("Citizens should be guaranteed freedom of choice in end-of-life matters (euthanasia).",
     "Citizens should be restricted from making end-of-life (euthanasia) choices."),

    ("Recreational use of marijuana/cannabis should be allowed.",
     "Recreational use of marijuana/cannabis should be prohibited."),

    ("A law is needed to prevent companies from relocating their production abroad.",
     "A law favoring businesses to relocate their production abroad is needed."),

    ("Businesses should be able to fire employees more easily.",
     "It should be more difficult for businesses to lay off employees."),

    ("The introduction of a single income tax rate (\"flat tax\") would benefit the Italian economy.",
     "The introduction of a single income tax rate (\"flat tax\") would be a disadvantage to the Italian economy."),

    ("An hourly minimum wage should be introduced.",
     "The hourly minimum wage should be ignored."),

    ("Italy should get out of the Eurozone.",
     "Italy should continue in the Eurozone."),

    ("The use of nuclear power plants for the purpose of producing energy should be promoted.",
     "The use of nuclear power plants for the purpose of producing energy should be limited."),

    ("The construction of Major Works is a priority for Italy.",
     "The construction of Major Works is irrelevant to Italy."),

    ("Regasifiers are necessary infrastructure for Italy.",
     "Regasifiers are irrelevant infrastructure for Italy."),

    ("Italy should keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO).",
     "Italy should cancel its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."),

    ("Italy should stop sending arms and war material to the Ukrainian government.",
     "Italy should continue sending arms and war material to the Ukrainian government."),

    ("Separation of careers between judges and prosecutors should be introduced.",
     "Career separation between judges and prosecutors is irrelevant."),

    ("The European Union should have a common foreign policy.",
     "The European Union should cancel the common foreign policy."),

    ("Direct election of the President of the Republic should be introduced.",
     "Direct election of the President of the Republic is irrelevant."),

    ("There should be a common European army.",
     "The effort for a common European army should cease."),

    ("Restrictions on personal freedom and privacy are acceptable to deal with health emergencies such as Covid-19.",
     "Restrictions on personal freedom and privacy are unacceptable to deal with health emergencies such as Covid-19."),

    ("Migrant landings must be stopped, even by extreme means.",
     "Migrant landings must continue, ceasing to resort to extreme means."),

    ("Children, born in Italy to foreign citizens and who have completed schooling should be granted Italian citizenship (ius scholae).",
     "Children, born in Italy to foreign nationals and who have completed schooling should have their Italian citizenship (ius scholae) denied."),

    ("More civil rights should be granted to homosexual, bisexual, transgender (LGBT+) people.",
     "Civil rights should be limited to homosexual, bisexual, transgender (LGBT+) people."),

    ("Organizers of events should be able to request a vaccination certificate upon entry.",
     "It should be impossible for event organizers to be able to request a vaccination certificate."),

    ("The government should abolish the ban on face-covering clothing.",
     "The government should keep the ban on face-covering clothing."),

    ("The Netherlands should build a new nuclear power plant.",
     "The Netherlands should scrap nuclear power plants."),

    ("Households with two partners, one of whom works, should receive the same tax benefits as households with two working partners.",
     "Households with two partners of which one works should receive less tax benefit as households with two working partners."),

    ("The Dutch government should apologize for the historical slave trade.",
     "The slave trade of the past is not the responsibility of the current Dutch government."),

    ("Citizens should have the opportunity to block laws passed by parliament through a referendum.",
     "Laws passed by parliament should not be stopped by a citizens' referendum."),

    ("Primary school teachers should earn as much as secondary school teachers.",
     "High school teachers should continue to earn more as elementary school teachers."),

    ("The Netherlands should spend more money on defense.",
     "The Netherlands should spend less money on defense."),

    ("Asylum seekers with a temporary residence permit should complete integration before getting a rental home.",
     "Asylum seekers with a temporary residence permit should be allowed to obtain rental housing before they are integrated."),

    ("Both the purchase and sale of soft drugs by coffee shops should be legalized.",
     "Both purchase and sale of soft drugs by coffee shops should remain illegal."),

    ("An increase in minimum wages should no longer automatically result in an increase in welfare benefits.",
     "Increasing minimum wages should automatically lead to increases in welfare benefits."),

    ("There should be no new restrictions on the activities of farming businesses.",
     "There should be new restrictions on farm activities."),

    ("A middle school should be established so that students make a choice between vocational education, general secondary education, or pre-university education at a later age.",
     "The arrival of a middle school is unnecessary, because students are now old enough when they have the choice between vmbo, havo or vwo."),

    ("The Netherlands should accept more refugees than it currently does.",
     "The Netherlands should accept fewer refugees than it does now."),

    ("People should always have the choice of whether to wear a face mask.",
     "Not the people themselves, but other agencies should choose whether or not to wear a mouth guard."),

    ("The Netherlands should exit the European Union (EU).",
     "The Netherlands must remain in the European Union (EU)."),

    ("Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists.",
     "A tax per kilometer driven should not replace the existing tax on car ownership for motorists."),

    ("During the upcoming New Year's Eve, it should be allowed to set off decorative fireworks again.",
     "Next turn of the year, it should remain illegal to set off decorative fireworks."),

    ("Less funding should go to public broadcasting.",
     "More money must go to public broadcasting."),

    ("The European Union should have less influence on Polish domestic policy.",
     "The European Union should have more influence on Polish domestic policy."),

    ("All entrepreneurs should pay the same health premium regardless of income.",
     "Entrepreneurs should pay health premiums based on their income."),

    ("The independence of the judiciary from parliament and the government should be strengthened.",
     "Parliamentary and government control over the judiciary should be strengthened."),

    ("The share of defense spending in Poland's GDP should be further increased.",
     "The share of defense spending in Poland's GDP should be reduced."),

    ("The powers of local governments should be increased at the expense of the central government.",
     "The powers of local governments should be reduced at the expense of the central government."),

    ("Poland should move away from coal mining no later than 2040.",
     "Poland should continue coal mining beyond 2040."),

    ("Poland should have grain imports from Ukraine blocked.",
     "Poland should support grain imports from Ukraine."),

    ("Public media funding from the state budget should be limited.",
     "Public media funding from the state budget should be increased."),

    ("Social transfers should be increased to reduce the effects of inflation on citizens.",
     "Social transfers should be reduced so as not to increase the burden on the state budget in times of inflation."),

    ("The powers of the secret services to track the activities of citizens on the Internet should be limited.",
     "The powers of the secret services to track citizens' activities on the Internet should be increased."),

    ("The state should finance private visits to specialists if the waiting time at a public facility exceeds three months.",
     "The state should refrain from financing private visits to specialists, regardless of the waiting time in a public facility."),

    ("The state should provide a free nursery place for every child.",
     "The state should refrain from providing free nursery places for all children."),

    ("Schools should have more freedom to choose the content covered in the curriculum.",
     "Schools should have less freedom to choose the content covered in the curriculum."),

    ("The state should build low-rent apartments for rent.",
     "The state should refrain from building low-income rental housing."),

    ("Taxes should be increased for top earners.",
     "Taxes should be lowered for top earners."),

    ("Early retirement should be introduced for those who have worked a certain number of years, regardless of their age.",
     "Early retirement based on the number of years worked, regardless of age, should be impossible."),

    ("Abortion laws should be liberalized.",
     "Abortion laws should be tightened."),

]

RULE_NAME = "Opposite"

TEMP_FOR_PROBS = 1.0
EPS = 1e-12

# For Appendix Fig A (head-combo ablation)
HEAD_COMBO_LAYER = 23
HEADS_TO_ABLATE = [1, 3, 6, 7]
HEAD_ABLATE_RATIO = 0.0


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
        raise RuntimeError("No Gemma3DecoderLayer found. Check transformers version/model class.")
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
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    enc = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    dev = get_input_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    answer_pos = input_ids.shape[-1] - 1

    digit_ids = []
    tok = processor.tokenizer
    for d in range(1, 8):
        ids = tok.encode(str(d), add_special_tokens=False)
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
def forward_logits_only(model: Gemma3ForConditionalGeneration, enc: EncodedChat) -> torch.Tensor:
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
    """
    p: [..., 7]，最后一维是 1..7 的概率。
    返回左右翻转后的分布（以 4 为中心镜像）。
    """
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


def collect_clean_cache(model: Gemma3ForConditionalGeneration, enc_clean: EncodedChat) -> CleanCache:
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
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_hook(i)))
            elif isinstance(sub, Gemma3MLP):
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
    model: Gemma3ForConditionalGeneration,
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
            if isinstance(sub, Gemma3Attention):
                hooks.append(sub.register_forward_hook(attn_patch_hook(i)))
            elif isinstance(sub, Gemma3MLP):
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
    model: Gemma3ForConditionalGeneration,
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
    model: Gemma3ForConditionalGeneration,
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
            if isinstance(sub, Gemma3Attention) and i in layers_to_edit:
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
            if isinstance(sub, Gemma3MLP) and i in layers_to_edit:
                hooks.append(sub.register_forward_hook(make_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Attention head COMBO ablation (only keep heads [1,3,6,7])
# ---------------------------
@contextlib.contextmanager
def attn_head_combo_ablation_context(
    model: Gemma3ForConditionalGeneration,
    enc: EncodedChat,
    layer_to_edit: int,
    heads_to_edit: List[int],
    ratio: float = 0.0,
):
    hooks = []
    num_heads = model.config.text_config.num_attention_heads

    def make_o_proj_hook(layer_idx: int):
        def _hook(module: nn.Linear, inputs, output):
            # Only intervene in the target layer
            if layer_idx != layer_to_edit:
                return output

            x = inputs[0]  # [B, T, H]
            B, T, H = x.shape
            head_dim = H // num_heads
            x = x.view(B, T, num_heads, head_dim)

            pos = enc.answer_pos
            for h_idx in heads_to_edit:
                if 0 <= h_idx < num_heads:
                    x[:, pos, h_idx, :] = x[:, pos, h_idx, :] * ratio

            x = x.view(B, T, H)

            # Recompute linear output
            W = module.weight
            b = module.bias
            out = torch.nn.functional.linear(x, W, b)
            return out
        return _hook

    for i, name, layer in get_decoder_layers(model):
        if i != layer_to_edit:
            continue
        for _, sub in layer.named_modules():
            if isinstance(sub, Gemma3Attention) and hasattr(sub, "o_proj"):
                hooks.append(sub.o_proj.register_forward_hook(make_o_proj_hook(i)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------
# Compute profiles for one pair
# ---------------------------
def patching_profile_for_pair(model, processor, base_text, variant_text):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    clean_cache = collect_clean_cache(model, enc_clean)
    layers = get_decoder_layers(model)
    n_layers = len(layers)

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

    block_arr = sweep("block")
    attn_arr = sweep("attn")
    mlp_arr = sweep("mlp")

    return {
        "block": block_arr,
        "attn": attn_arr,
        "mlp": mlp_arr,
        "n_layers": n_layers,
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
    }


def ablation_profile_for_pair(model, processor, base_text, variant_text, ratio: float = 0.0):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

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


def head_combo_effect_for_pair(model, processor, base_text, variant_text):
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    # Apply head-combo ablation at the chosen layer
    with attn_head_combo_ablation_context(
        model=model,
        enc=enc_corrupt,
        layer_to_edit=HEAD_COMBO_LAYER,
        heads_to_edit=HEADS_TO_ABLATE,
        ratio=HEAD_ABLATE_RATIO,
    ):
        logits_ab = forward_logits_only(model, enc_corrupt)
        ab_probs = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

    r = normalized_restoration(w_1d, clean_probs, corrupt_probs, ab_probs)

    return {
        "clean_probs": clean_probs.detach().float().cpu().numpy(),
        "corrupt_probs": corrupt_probs.detach().float().cpu().numpy(),
        "ab_probs": ab_probs.detach().float().cpu().numpy(),
        "restoration": float(r.item()),
    }


def head_sweep_restoration_for_pair(
    model,
    processor,
    base_text: str,
    variant_text: str,
    layer: int = 23,
    ratio: float = 0.0
):
    """
    Sweep all heads at `layer`, ablate ONE head at a time (ratio=0.0 by default),
    return per-head restoration in head-index order (0..num_heads-1).

    Returns:
        head_ids: List[int] = [0,1,...,H-1]
        head_restorations: np.ndarray shape [H]
        combo_restoration: float restoration for HEADS_TO_ABLATE combo
    """
    enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(base_text))
    enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(variant_text))

    logits_clean = forward_logits_only(model, enc_clean)
    logits_corrupt = forward_logits_only(model, enc_corrupt)

    clean_probs = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
    corrupt_probs = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

    num_heads = model.config.text_config.num_attention_heads

    head_ids = list(range(num_heads))
    head_restorations = np.full((num_heads,), np.nan, dtype=np.float64)

    # one-head-at-a-time
    for h in range(num_heads):
        with attn_head_combo_ablation_context(
            model=model,
            enc=enc_corrupt,
            layer_to_edit=layer,
            heads_to_edit=[h],
            ratio=ratio,
        ):
            logits_ab = forward_logits_only(model, enc_corrupt)
            probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

        r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
        head_restorations[h] = float(r.item())

    # combo heads [1,3,6,7] (your requirement)
    with attn_head_combo_ablation_context(
        model=model,
        enc=enc_corrupt,
        layer_to_edit=layer,
        heads_to_edit=HEADS_TO_ABLATE,
        ratio=ratio,
    ):
        logits_ab = forward_logits_only(model, enc_corrupt)
        probs_ab = digit_probs_from_logits_full(logits_ab, enc_corrupt, TEMP_FOR_PROBS)

    combo_r = normalized_restoration(w_1d, clean_probs, corrupt_probs, probs_ab)
    combo_restoration = float(combo_r.item())

    return head_ids, head_restorations, combo_restoration


def head_sweep_restoration_mean_over_pairs(
    model,
    processor,
    pairs: List[Tuple[str, str]],
    layer: int = 23,
    ratio: float = 0.0,
):
    """
    Compute per-head restoration for each pair, then average over all pairs.
    Returns head_ids, mean_head_restorations, mean_combo_restoration.
    """
    all_head_rest = []
    all_combo_rest = []

    # compute per-pair
    for i, (b, v) in enumerate(pairs, 1):
        print(f"  Head sweep pair {i}/{len(pairs)}")
        head_ids, head_rest, combo_r = head_sweep_restoration_for_pair(
            model=model,
            processor=processor,
            base_text=b,
            variant_text=v,
            layer=layer,
            ratio=ratio,
        )
        all_head_rest.append(head_rest)
        all_combo_rest.append(combo_r)

    # stack: [N, H] -> mean over N
    mat = np.stack(all_head_rest, axis=0)
    mean_head_rest = np.nanmean(mat, axis=0)

    mean_combo = float(np.nanmean(np.array(all_combo_rest, dtype=np.float64)))

    return head_ids, mean_head_rest, mean_combo


# ---------------------------
# Aggregation
# ---------------------------
def bootstrap_ci_median(data_2d: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    N, L = data_2d.shape
    med = np.nanmedian(data_2d, axis=0)

    boot = np.empty((n_boot, L), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot[b] = np.nanmedian(data_2d[idx], axis=0)

    lo = np.nanpercentile(boot, 100 * (alpha / 2), axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return med, lo, hi


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



def plot_head_bars_by_index(
    title: str,
    head_ids: List[int],
    head_restorations: np.ndarray,
    combo_restoration: float,
    layer: int,
):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.2))  # 宽一点，不然标签挤
    x = np.array(head_ids, dtype=int)

    ax.bar(x, head_restorations)

    ax.set_title(
        f"{title}\nLayer {layer} per-head ablation restoration (by head index)\n"
        f"Combo heads {HEADS_TO_ABLATE} restoration: {combo_restoration:.3f}"
    )
    ax.set_xlabel("Head index")
    ax.set_ylabel("Normalized restoration score")

    # 关键：标出所有 head 的刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in head_ids], rotation=90, fontsize=7)

    # 组合 ablation 的水平虚线（作为对照）
    ax.axhline(combo_restoration, linestyle="--")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig




def plot_digit_probs_clean_corrupt_ablate(title: str, clean_probs, corrupt_probs, ab_probs, restoration_value: float):
    digits = np.arange(1, 8)
    width = 0.26

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    ax.bar(digits - width, clean_probs, width=width, label="Clean (base)")
    ax.bar(digits, corrupt_probs, width=width, label="Corrupt (variant)")
    ax.bar(digits + width, ab_probs, width=width, label=f"Head-ablate L{HEAD_COMBO_LAYER} heads {HEADS_TO_ABLATE}")

    ax.set_title(f"{title}\nNormalized restoration (W1-based): {restoration_value:.3f}")
    ax.set_xlabel("Likert digit")
    ax.set_ylabel("Probability")
    ax.set_xticks(digits)
    ax.set_ylim(0.0, max(clean_probs.max(), corrupt_probs.max(), ab_probs.max()) * 1.15 + 1e-6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


# ---------------------------
# Main
# ---------------------------
def main():
    set_global_determinism(0, single_thread=True)
    torch.set_grad_enabled(False)

    print("Loading model + processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    # ---------------------------
    # Appendix Fig A (one subplot): Head-combo ablation at layer 23 (heads 1,3,6,7)
    # For simplicity (and interpretability), we plot ONE representative pair:
    # the first pair in PAIRS.
    # ---------------------------
    # print(f"\n[Appendix Fig A] Per-head ablation bar chart (by head index) at layer {HEAD_COMBO_LAYER}")
    
    # head_ids, head_restorations, combo_r = head_sweep_restoration_mean_over_pairs(
    #     model=model,
    #     processor=processor,
    #     pairs=PAIRS,                 # <-- 关键：用全部 pairs
    #     layer=HEAD_COMBO_LAYER,
    #     ratio=HEAD_ABLATE_RATIO,
    # )


    # figA = plot_head_bars_by_index(
    #     title=f"Attention head ablation\n{RULE_NAME}",
    #     head_ids=head_ids,
    #     head_restorations=head_restorations,
    #     combo_restoration=combo_r,
    #     layer=HEAD_COMBO_LAYER,
    # )
    # figA.savefig("opposite_head_bars_by_index.png", dpi=200)
    # print("Saved: appendix_figA_head_bars_by_index.png")

    # ---------------------------
    # Figure 1 (one subplot): Activation patching curves
    # ---------------------------
    print(f"\n[Figure 1] Activation patching profiles for rule: {RULE_NAME}")
    patch_profiles = []
    for i, (b, v) in enumerate(PAIRS, 1):
        print(f"  Patching pair {i}/{len(PAIRS)}")
        patch_profiles.append(patching_profile_for_pair(model, processor, b, v))
    patch_stats = aggregate_profiles(patch_profiles)

    fig1 = plot_layerwise_subplot(
        title=f"Activation patching\n{RULE_NAME} (unflip pairs n={patch_stats['n_pairs']})",
        stats=patch_stats,
        ylabel="Normalized restoration score",
    )
    fig1.savefig("opposite_patching_one_rule.png", dpi=200)
    print("Saved: fig1_patching_one_rule.png")

    # ---------------------------
    # Figure 2 (one subplot): Ablation curves (ratio=0.0)
    # ---------------------------
    # print(f"\n[Figure 2] Ablation (ratio=0.0) profiles for rule: {RULE_NAME}")
    # ab_profiles = []
    # for i, (b, v) in enumerate(PAIRS, 1):
    #     print(f"  Ablation pair {i}/{len(PAIRS)}")
    #     ab_profiles.append(ablation_profile_for_pair(model, processor, b, v, ratio=0.0))
    # ab_stats = aggregate_profiles(ab_profiles)

    # fig2 = plot_layerwise_subplot(
    #     title=f"Inference-time masking (ratio=0.0)\n{RULE_NAME} (flip pairs n={ab_stats['n_pairs']})",
    #     stats=ab_stats,
    #     ylabel="Normalized restoration score",
    # )
    # fig2.savefig("opposite_ablation_one_rule.png", dpi=200)
    # print("Saved: fig2_ablation_one_rule.png")

    # Show all three figures
    plt.show()


if __name__ == "__main__":
    main()
