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
#     ("There should be an additional tax on purchasing meat.",
#      "There should be a no additional tax on buying meat."),

#     ("The national government, rather than provinces and municipalities, should decide where new residential areas are built.",
#      "The national government should not decide where new residential areas are built instead of provinces and municipalities."),

#     ("Houses should be built on land currently used for agriculture.",
#      "No housing should be built on land now used for agriculture."),

#     ("There should be fewer options for community service sentences instead of prison sentences.",
#      "There should not be fewer opportunities to impose community service instead of prison sentences."),

#     ("People who consider their lives complete should be able to receive assistance with suicide.",
#      "People who consider their lives complete should not be able to get help with suicide."),

#     ("An increase in minimum wages should no longer automatically result in an increase in welfare benefits.",
#      "Raising minimum wages should still automatically increase welfare payments."),

#     ("A middle school should be established so that students make a choice between vocational education, general secondary education, or pre-university education at a later age.",
#      "There should be no middle school, so that pupils do not have to choose between vmbo, havo or vwo at a later age."),

#     ("Limiting rights and freedoms is necessary to combat organized crime.",
#      "It is not necessary to limit rights and freedoms to combat organized crime."),

#     ("The growth of Islam is a threat to Spain's security.",
#      "The growth of Islam is not a threat to Spain's security."),

#     ("The efficiency of public services improves when they are privatized.",
#      "Efficiency in the provision of public services does not improve when they are privatized."),

#     ("It should be easier for companies to fire workers.",
#      "It should not be easier for companies to lay off workers."),

#     ("Climate change is solely attributable to human action.",
#      "Climate change is not solely attributable to human action."),

#     ("The future Spanish government should increase irrigated agricultural areas by means of large water transfers.",
#      "The future Spanish government should not increase agricultural irrigated areas through large water transfers."),

#     ("Negotiating with pro-independence supporters weakens the State.",
#      "Negotiating with the independentistas does not weaken the State."),

#     ("The policies of linguistic immersion in the native language of bilingual Autonomous Communities endanger Spanish.",
#      "The policies of linguistic immersion in the language of the bilingual Autonomous Communities do not endanger Spanish."),

#     ("It is necessary to repeal the Law of Democratic Memory passed during this legislature.",
#      "It is not necessary to repeal the Law of Democratic Memory passed during this legislature."),

#     ("Immigrants should pay for their own health services.",
#      "Immigrants should not have to pay for their health services."),

#     ("Covid-19 vaccines are to continue to be protected by patents.",
#      "Vaccines against Covid-19 should not continue to be protected by patents."),

#     ("The traditional family of father, mother and children is to be promoted more strongly than other living arrangements.",
#      "The traditional family of father, mother and children should not be promoted more than other cohabiting couples."),

#     ("Students should receive BAföG regardless of their parents' income.",
#      "Students should receive BAföG depending on their parents' income."),

#     ("The Nord Stream 2 Baltic Sea pipeline, which transports gas from Russia to Germany, is to be allowed to go into operation as planned.",
#      "The \"\"Nord Stream 2\"\" Baltic Sea pipeline, which transports gas from Russia to Germany, should not be allowed to go into operation as planned."),

#     ("The registration of new cars with combustion engines should also be possible in the long term.",
#      "The registration of new cars with combustion engines should no longer be possible in the long term."),

#     ("The state should continue to collect church tax for religious communities.",
#      "The state should not collect church tax for religious communities."),

#     ("Inpatient treatment in hospitals is to continue to be charged on the basis of a flat rate per case.",
#      "Inpatient treatment in hospitals should not be charged at a flat rate per case."),

#     ("A tax is to be levied again on high assets.",
#      "No tax should be levied on high assets."),

#     ("Facial recognition software should be allowed to be used for video surveillance in public places.",
#      "No facial recognition software should be used for video surveillance in public places."),

#     ("Air traffic is to be taxed more heavily.",
#      "Air traffic should not be taxed more heavily."),

#     ("The European Union should have less influence on Polish domestic policy.",
#      "The European Union should not have less influence on Polish domestic policy."),

#     ("The state should finance private visits to specialists if the waiting time at a public facility exceeds three months.",
#      "The state should not finance private visits to specialists if the waiting time at a public facility exceeds three months."),

#     ("The result of any nationwide referendum should be binding regardless of turnout.",
#      "The results of certain nationwide referendums should be binding depending on turnout."),

#     ("Poland should adopt the migrant relocation solutions adopted by the European Union.",
#      "Poland should not adopt the migrant relocation solution adopted by the European Union."),

#     ("The powers of local governments should be increased at the expense of the central government.",
#      "The powers of local governments should not be increased at the expense of the central government."),

#     ("Christian values should be the basis of state social policy.",
#      "Christian values should not be the basis of state social policy."),

#     ("The EU's rule of law mechanism threatens Hungary's sovereignty.",
#      "The EU's rule of law mechanism does not threaten Hungary's sovereignty."),

#     ("Stronger state regulation of the work of NGOs supported by foreign organisations is needed.",
#      "There is no need for stronger state regulation of the work of NGOs supported by foreign organisations."),

#     ("Political influence has been reduced by changing the university model (reorganisation into a trust).",
#      "The change in the university model (reorganisation into a trust) has not reduced political influence."),

#     ("One effective way to reduce rents is to conclude favourable gas supply contracts with Russia.",
#      "The conclusion of favourable gas supply contracts with Russia is not an effective way of reducing rationing."),

#     ("A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is the right step to fight inflation.",
#      "A price freeze on some basic foodstuffs (e.g. chicken tail, milk) is not the right way to fight inflation."),

#     ("Migrant landings must be stopped, even by extreme means.",
#      "Migrant landings must not be stopped, even by extreme means."),

#     ("A heritage tax one's wealth should be introduced.",
#      "A wealth tax on great wealth should not be introduced."),

#     ("The citizenship allowance is a measure that should be cancelled.",
#      "The citizenship income is not a measure to be cancelled."),

#     ("Beach concessions to private individuals should be time-limited.",
#      "Beach concessions to private individuals should not be time-limited."),

#     ("Italy should build more incinerators/thermal power plants.",
#      "Italy should not build more incinerators/thermal power plants."),

#     ("Drilling is necessary to find more energy resources.",
#      "Drilling is not necessary to find more energy resources."),

#     ("Taxes collected at the regional level should be fully retained in the region itself.",
#      "Taxes collected at the regional level should not be fully retained in the region itself."),

#     ("Compulsory vaccination of children should be introduced based on the Swiss vaccination plan.",
#      "Compulsory vaccination of children should not be introduced based on the Swiss vaccination plan."),

#     ("Insured persons should contribute more to health care costs (e.g., increase the minimum deductible).",
#      "Insured persons should not contribute more to health care costs (e.g., increase the minimum deductible)."),

#     ("The federal government should be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services).",
#      "The federal government should not be given the authority to determine the hospital offering (national hospital planning with regard to locations and range of services)."),

#     ("The federal government should raise the requirements for the high school.",
#      "The federal government should not raise the requirements for the high school."),

#     ("A third official gender should be introduced alongside \"\"female\"\" and \"\"male\"\".",
#      "A third official gender should not be introduced alongside \"\"female\"\" and \"\"male\"\"."),

#     ("To achieve climate targets, incentives and target agreements should be relied on exclusively, rather than bans and restrictions.",
#      "To achieve climate targets, incentives and target agreements should not be relied on exclusively, rather than bans and restrictions."),

#     ("The army's target number of soldiers should expand to at least 120,000.",
#      "The army's target number of soldiers should not expand to at least 120,000."),

#     ("The Federal Council should be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine).",
#      "The Federal Council should not be allowed to authorize other states to re-export Swiss weapons in cases of a war of aggression in violation of international law (e.g., the attack on Ukraine)."),

#     ("Switzerland should terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons.",
#      "Switzerland should not terminate the Bilateral Agreements with the EU and seek a free trade agreement without the free movement of persons."),
# ]

PAIRS: List[Tuple[str, str]] = [
    ("There should be a ban on single-use plastic and non-recyclable plastics. ", "There should not be a ban on single-use plastic and non-recyclable plastics."),
    ("There should be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). ", "There should not be government measures to make the use of electronic devices more sustainable (e.g., right to repair, extension of warranty period, minimum guaranteed period for software updates). "),
    ("Switzerland should terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border. ", "Switzerland should not terminate the Schengen agreement with the EU and reintroduce more security checks directly on the border."),
    ("There should be closer relations with the European Union (EU). ", "There should not be closer relations with the European Union (EU)."),
    ("Switzerland should strive for a comprehensive free trade agreement (including agriculture) with the USA. ", "Switzerland should not strive for a comprehensive free trade agreement (including agriculture) with the USA."),
    ("Companies should be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards. ", "Companies should not be obliged to ensure that their subsidiaries and suppliers operating abroad comply with social and environmental standards."),
    ("Switzerland should return to a strict interpretation of neutrality (renounce economic sanctions to a large extent). ", "Switzerland should not return to a strict interpretation of neutrality (renounce economic sanctions to a large extent)."),
    ("There should be an increase in the retirement age. ", "There should not be an increase in the retirement age."),
    ("The federal government should allocate more funding for health insurance premium subsidies. ", "The federal government should not allocate more funding for health insurance premium subsidies."),
    ("The Swiss mobile network should be equipped throughout the country with the latest technology (currently 5G standard). ", "The Swiss mobile network should not be equipped throughout the country with the latest technology (currently 5G standard)."),
    ("For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should be eliminated.", "For married couples, the pension is currently limited to 150% of the maximum individual AHV pension (capping). This limit should not be eliminated."),
    ("As part of the reform of the BVG (occupational pension plan), pensions are to be reduced (lowering the minimum conversion rate from 6.8% to 6%). ", "As part of the reform of the BVG (occupational pension plan), pensions are not to be reduced (lowering the minimum conversion rate from 6.8% to 6%). "),
    ("Paid parental leave should be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave. ", "Paid parental leave should not be increased beyond today's 14 weeks of maternity leave and two weeks of paternity leave."),
    ("The federal government should provide more financial support for public housing construction. ", "The federal government should not provide more financial support for public housing construction."),
    ("There should be an introduction of a tax on foods containing sugar (sugar tax). ", "There should not be an introduction of a tax on foods containing sugar (sugar tax)."),
    ("The Federal Council's ability to restrict private and economic life in the event of a pandemic should be more limited. ", "The Federal Council's ability to restrict private and economic life in the event of a pandemic should not be more limited."),
    ("According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should be taught in regular classes. ", "According to the Swiss integrated schooling concept, children with learning difficulties or disabilities should not be taught in regular classes. "),
    ("The federal government should be given additional powers in the area of digitization of government services in order to be able to impose binding directives and standards on the cantons.", "The federal government should not be given additional powers in the area of digitization of government services in order to be able to impose binding directives and standards on the cantons."),
    ("The state should be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). ", "The state should not be more committed to equal educational opportunities (e.g., through subsidized remedial courses for students from low-income families). "),
    ("The conditions for naturalization should be relaxed (e.g., shorter residence period). ", "The conditions for naturalization should not be relaxed (e.g., shorter residence period). "),
    ("More qualified workers from non-EU/EFTA countries should be allowed to work in Switzerland (increase third-country quota).", "More qualified workers from non-EU/EFTA countries should not be allowed to work in Switzerland (increase third-country quota)."),
    ("Foreign nationals who have lived in Switzerland for at least ten years should be granted the right to vote and stand for election at the municipal level. ", "Foreign nationals who have lived in Switzerland for at least ten years should not be granted the right to vote and stand for election at the municipal level."),
    ("Cannabis use should be legalized. ", "Cannabis use should not be legalized."),
    ("Doctors should be allowed to administer direct active euthanasia. ", "Doctors should not be allowed to administer direct active euthanasia."),
    ("Same-sex couples should have the same rights as heterosexual couples in all areas. ", "Same-sex couples should not have the same rights as heterosexual couples in all areas."),
    ("There should be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). ", "There should not be a stronger regulation of the major Internet platforms (i.e., transparency rules on algorithms, increased liability for content, combating disinformation). "),
    ("There should be tax cuts at the federal level over the next four years. ", "There should not be tax cuts at the federal level over the next four years."),
    ("The differences between cantons with high and low financial capacity should be further reduced through fiscal equalization. ", "The differences between cantons with high and low financial capacity should not be further reduced through fiscal equalization."),
    ("A minimum wage of CHF 4,000 for all full-time employees should be introduced. ", "A minimum wage of CHF 4,000 for all full-time employees should not be introduced. "),
    ("There should be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). ", "There should not be stricter regulations for the financial sector (e.g., stricter capital requirements for banks, ban on bonuses). "),
    ("Private households should be free to choose their electricity supplier (complete liberalization of the electricity market). ", "Private households should not be free to choose their electricity supplier (complete liberalization of the electricity market)."),
    ("There should be stricter controls on equal pay for women and men. ", "There should not be stricter controls on equal pay for women and men."),
    ("Busy sections of highways should be widened. ", "Busy sections of highways should not be widened."),
    ("There should be a popular initiative aims to reduce television and radio fees (CHF 200 per household, exemption for businesses).", "There should not be a popular initiative aims to reduce television and radio fees (CHF 200 per household, exemption for businesses)."),
    ("Switzerland should ban the registration of new passenger cars with internal combustion engines starting in 2035. ", "Switzerland should not ban the registration of new passenger cars with internal combustion engines starting in 2035."),
    ("The construction of new nuclear power plants should be allowed again. ", "The construction of new nuclear power plants should not be allowed again."),
    ("The state should guarantee a comprehensive public service offering also in rural regions. ", "The state should not guarantee a comprehensive public service offering also in rural regions."),
    ("Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should be introduced. ", "Increasing electricity tariffs when consumption is higher (progressive electricity tariffs) should not be introduced."),
    ("The protection regulations for large predators (lynx, wolf, bear) should be relaxed. ", "The protection regulations for large predators (lynx, wolf, bear) should not be relaxed. "),
    ("Direct payments should only be granted to farmers with proof of ecological performance. ", "Direct payments should not only be granted to farmers with proof of ecological performance."),
    ("There should be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas). ", "There should not be stricter animal welfare regulations for livestock (e.g. permanent access to outdoor areas)."),
    ("30% of Switzerland's land area should be dedicated to preserving biodiversity?. ", "30% of Switzerland's land area should not be dedicated to preserving biodiversity. "),
    ("The voting age should be lowered to 16. ", "The voting age should not be lowered to 16."),
    ("It should be possible to hold a referendum on federal spending above a certain amount (optional financial referendum). ", "It should not be possible to hold a referendum on federal spending above a certain amount (optional financial referendum)."),
    ("The Swiss Armed Forces should expand their cooperation with NATO. ", "The Swiss Armed Forces should not expand their cooperation with NATO."),
    ("A general speed limit is to apply on all highways.", "No general speed limit should apply on all highways."),
    ("Donations from companies to political parties should continue to be permitted.", "Donations from companies to political parties should not be permitted."),
    ("In Germany, it should generally be possible to have a second citizenship in addition to the German one.", "In Germany, it should generally not be possible to have a second citizenship in addition to German citizenship."),
    ("Federal authorities are to take linguistic account of different gender identities in their publications.", "Federal authorities should not use different gender identities in their publications."),
    ("Female civil servants are to be allowed to wear headscarves while on duty.", "Female civil servants should not be allowed to wear headscarves on duty."),
    ("Germany is to increase its defense spending.", "Germany should not increase its defense spending."),
    ("The federal government is to be given more responsibilities in school policy.", "The federal government should not be given more responsibilities in school policy."),
    ("The federal government is to provide more financial support for projects to combat anti-Semitism.", "The federal government should not provide more financial support for projects to combat anti-Semitism."),
    ("Chinese companies should not be allowed to receive contracts for the expansion of the communications infrastructure in Germany.", "Chinese companies should be allowed to receive contracts for the expansion of the communications infrastructure in Germany."),
    ("The controlled sale of cannabis is to be generally permitted.", "The controlled sale of cannabis should not be permitted."),
    ("Germany is to leave the European Union.", "Germany should not leave the European Union."),
    ("The state lists of the parties for the elections to the German Bundestag are to have to be filled alternately by women and men.", "The state lists of the parties for the elections to the German Bundestag should not have to be filled alternately by women and men."),
    ("Young people over the age of 16 are to be allowed to vote in Bundestag elections.", "Young people aged 16 and over should not be allowed to vote in federal elections."),
    ("Married couples without children should continue to receive tax breaks.", "Married couples without children should not continue to receive tax breaks."),
    ("Organic agriculture should be promoted more strongly than conventional agriculture.", "Organic farming should not be promoted more than conventional farming."),
    ("Islamic associations are to be able to be recognized by the state as religious communities.", "Islamic associations should not be able to be recognized by the state as religious communities."),
    ("The government-set price for CO2 emissions from heating and driving is to rise more than planned.", "The state-fixed price for CO2 emissions from heating and driving should not increase more than planned."),
    ("The debt brake in the Basic Law is to be retained.", "The debt brake in the Basic Law should not be retained."),
    ("Asylum is to continue to be granted only to politically persecuted persons.", "Asylum should not only be granted to politically persecuted persons."),
    ("The statutory minimum wage is to be increased to at least 12 euros by 2022 at the latest.", "The statutory minimum wage should not be increased to at least 12 euros by 2022 at the latest."),
    ("Subsidies for wind energy are to be ended.", "The promotion of wind energy should not be ended."),
    ("The ability of landlords to increase housing rents is to be more strictly limited by law.", "The ability of landlords to increase rents should not be more strictly limited by law."),
    ("The phase-out of coal-fired power generation planned for 2038 is to be brought forward.", "The phase-out of coal-fired power generation planned for 2038 should not be brought forward."),
    ("All employed persons are to be required to be insured in the statutory pension scheme.", "People in employment should not necessarily have to be insured under the statutory pension scheme."),
    ("The right of recognized refugees to join their families is to be abolished.", "The right of recognized refugees to family reunification should not be abolished."),
    ("Governments should intervene as little as possible in the economy.", "Governments should not intervene as little as possible in the economy."),
    ("Taxes on fossil fuels must be raised to finance the Green Transition.", "Taxes on fossil fuels should not be raised to finance the Ecological Transition."),
    ("To better defend Spain's interests in Europe we must recover more sovereignty.", "In order to better defend Spain's interests in Europe, we should not recover more sovereignty."),
    ("Spanish government should promote the strengthening of NATO in Europe.", "The Spanish government should not promote the strengthening of NATO in Europe."),
    ("The best way to solve the conflict in Catalonia is for its citizens to be able to vote on their future in a referendum.", "The best way to solve the conflict in Catalonia is that its citizens cannot vote on their future in a referendum."),
    ("Spain's territorial decentralization must be deepened.", "There is no need to deepen the territorial decentralization of Spain."),
    ("The right to self-determination must be recognized by the Constitution.", "The right of self-determination should not be recognized by the Constitution."),
    ("Spain should be more tolerant with illegal migration.", "Spain should not be more tolerant of illegal immigration."),
    ("Housing prices must be regulated to ensure access for all people.", "Housing prices should not be regulated to guarantee access to all people."),
    ("Current gender policies are biased against men.", "Current gender policies are not against men."),
    ("The state should take measures to redistribute wealth from the rich to the poor.", "The state should not take measures to redistribute wealth from the rich to the poor."),
    ("The government must increase spending on public health care, even if this means increasing taxes.", "The government should not increase spending on the public health system even if this means increasing taxes."),
    ("A permanent tax on large fortunes and assets is necessary.", "There is no need for a permanent tax on large fortunes and wealth."),
    ("The working day should be reduced without reducing workers' wages.", "Working hours should not be reduced without reducing workers' wages."),
    ("Education spending should be increased to at least the OECD average of 5.2 per cent (GDP).", "Spending on education should not be increased to the OECD average of 5.2 per cent (of GDP)."),
    ("Teachers' salaries should be doubled.", "Teachers' salaries should not be doubled."),
    ("Only men and women should be allowed to marry.", "Marriages should not be exclusively between men and women."),
    ("Parties should strive for a closer ratio of men to women when drawing up lists.", "Parties should not seek to have a close ratio of men to women when drawing up lists."),
    ("The state should take targeted measures to promote equal participation of fathers and mothers in child-rearing.", "The state should not take targeted measures to encourage fathers and mothers to share equally in child-rearing."),
    ("The Hungarian government should ratify the Istanbul Convention, which combats violence against women and domestic violence.", "The Hungarian government should not ratify the Istanbul Convention against violence against women and domestic violence."),
    ("Comprehensive public procurement reform is needed (e.g. opening up large-scale centralised public procurement to smaller firms).", "There is no need for comprehensive public procurement reform (e.g. opening up large-scale centralised public procurement to smaller firms)."),
    ("Increase the contribution of the wealthier to the public purse (abolition of the one-band tax).", "The wealthier should not contribute more to the public burden (abolition of the one-band tax)."),
    ("Hungarian foreign policy should be guided solely by Hungarian economic interests.", "Hungarian foreign policy should not be driven solely by Hungarian economic interests."),
    ("Public employment helps people re-enter the labour market.", "Public works do not help people to re-enter the labour market."),
    ("State regulation of the rental housing market is not necessary.", "Public regulation of the rental housing market is needed."),
    ("The current three-month unemployment benefit should be extended.", "An extension of the current three-month unemployment benefit is not necessary."),
    ("A family tax credit is a better way to support families than increasing the family allowance.", "The family tax allowance is no better way of supporting families than increasing the family allowance."),
    ("The use of medical cannabis should be legalised in Hungary.", "Medical cannabis should not be legalised in Hungary."),
    ("An independent Ministry of Health should be established.", "There should be no separate Ministry of Health."),
    ("Hungary should decide by referendum whether to remain part of the EU.", "Hungary should not decide by referendum whether to remain part of the EU."),
    ("Comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner-take-all compensation, extension of postal voting) is needed.", "There is no need for a comprehensive reform of the electoral system (redrawing of district boundaries, abolition of winner compensation, extension of postal voting)."),
    ("A legal framework for primary elections should be provided.", "There is no need to provide a legal framework for primary elections."),
    ("Voting age for elections should be 16.", "Voting age should not be 16 or older."),
    ("Internet access should be free for all.", "Internet should not be free for all."),
    ("Polluting companies should be taxed more heavily.", "Polluting companies should not be subject to higher taxes."),
    ("In larger cities, car traffic should be limited through various measures (P+R parking, construction of cycle paths, improvement of public transport).", "In larger cities, there is no need to restrict car traffic through various measures (P+R parking, building cycle paths, improving public transport)."),
    ("The redevelopment of urban green spaces (e.g. the Liget project in Budapest) needs a broad social dialogue.", "The redevelopment of urban green areas (e.g. the Liget project in Budapest) does not require a broad social dialogue."),
    ("An independent ministry for the environment is needed.", "There is no need for a separate environment ministry."),
    ("Gender identity can be influenced by environmental influences (e.g. media content, sensitising activities).", "Gender identity should not be influenced by environmental influences (e.g. media content, sensitising activities)."),
    ("An animal rights commissioner should be introduced.", "No need to introduce an animal rights commissioner."),
    ("Hungary should join the European Public Prosecutor's Office.", "Hungary should not join the European Public Prosecutor's Office."),
    ("Stricter regulation of interception software (e.g. Pegasus) is needed (e.g. subject to judicial authorisation).", "There is no need for stricter regulation of interception software (e.g. Pegasus) (e.g. subject to judicial authorisation)."),
    ("Disclosure of the origin of criminals is needed for more effective law enforcement.", "Disclosing the origin of criminals is not necessary for more effective law enforcement."),
    ("The age of compulsory schooling should be raised back to 18.", "The age of compulsory education should not be raised back to 18."),
    ("European integration is all in all a positive process.", "European integration is not an all-positive process."),
    ("Citizens should be guaranteed freedom of choice in end-of-life matters (euthanasia).", "Citizens should not be guaranteed freedom of choice in end-of-life matters (euthanasia)."),
    ("Recreational use of marijuana/cannabis should be allowed.", "Recreational use of marijuana/cannabis should not be allowed."),
    ("A law is needed to prevent companies from relocating their production abroad.", "A law preventing businesses from relocating their production abroad is not needed."),
    ("Businesses should be able to fire employees more easily.", "Businesses should not be allowed to lay off employees more easily."),
    ("Health care should be managed only by the state and not by private individuals.", "Health care should not only be managed by the state, but also by private individuals."),
    ("The introduction of a single income tax rate (\"flat tax\") would benefit the Italian economy.", "The introduction of a single income tax rate (\"flat tax\") would not benefit the Italian economy."),
    ("An hourly minimum wage should be introduced.", "The hourly minimum wage should not be introduced."),
    ("Italy should get out of the Eurozone.", "Italy should not leave the Euro."),
    ("The use of nuclear power plants for the purpose of producing energy should be promoted.", "The use of nuclear power plants for the purpose of producing energy should not be promoted."),
    ("The construction of Major Works is a priority for Italy.", "The construction of Major Works is not a priority for Italy."),
    ("Regasifiers are necessary infrastructure for Italy.", "Regasifiers are not necessary infrastructure for Italy."),
    ("Italy should keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO).", "Italy should not keep its foreign policy aligned with the choices of the Atlantic Alliance (NATO)."),
    ("Sanctions against Russia should be tougher.", "Sanctions against Russia should not be tougher."),
    ("Italy should stop sending arms and war material to the Ukrainian government.", "Italy should not stop sending arms and war material to the Ukrainian government."),
    ("The European Union should have a common foreign policy.", "The European Union should not have a common foreign policy."),
    ("Direct election of the President of the Republic should be introduced.", "Direct election of the President of the Republic should not be introduced."),
    ("There should be a common European army.", "There should not be a common European army."),
    ("European economic integration has gone too far: member states should regain more autonomy.", "European economic integration has not gone too far: member states should not regain more autonomy."),
    ("Restrictions on personal freedom and privacy are acceptable to deal with health emergencies such as Covid-19.", "Restrictions on personal freedom and privacy are not acceptable for dealing with health emergencies such as Covid-19."),
    ("Children, born in Italy to foreign citizens and who have completed schooling should be granted Italian citizenship (ius scholae).", "Children, born in Italy to foreign nationals and who have completed school, should not be granted Italian citizenship (ius scholae)."),
    ("More civil rights should be granted to homosexual, bisexual, transgender (LGBT+) people.", "Homosexual, bisexual, transgender (LGBT+) people should not be granted more civil rights."),
    ("Organizers of events should be able to request a vaccination certificate upon entry.", "Event organizers should not require vaccination certificates at entry."),
    ("The government should abolish the ban on face-covering clothing.", "The government should not abolish the ban on face-covering clothing."),
    ("The government should reduce the VAT on cultural activities to 5 percent.", "The government should not reduce VAT on cultural activities to 5 percent."),
    ("The Netherlands should build a new nuclear power plant.", "The Netherlands should not build a new nuclear power plant."),
    ("Households with two partners, one of whom works, should receive the same tax benefits as households with two working partners.", "Households with two partners of which one works should not receive the same tax benefit as households with two working partners."),
    ("The Dutch government should apologize for the historical slave trade.", "The Dutch government should not apologize for the slave trade in the past."),
    ("Citizens should have the opportunity to block laws passed by parliament through a referendum.", "Citizens should not be allowed to stop laws passed by parliament through a referendum."),
    ("Primary school teachers should earn as much as secondary school teachers.", "Teachers in elementary schools should not start earning as much as teachers in secondary schools."),
    ("The Netherlands should spend more money on defense.", "The Netherlands should not spend more money on defense."),
    ("The Netherlands should introduce an additional flight tax for short-distance flights.", "The Netherlands should not introduce an additional air tax for short-haul flights."),
    ("Asylum seekers with a temporary residence permit should complete integration before getting a rental home.", "Asylum seekers with provisional residence permits should not have to integrate before they are given rental housing."),
    ("Both the purchase and sale of soft drugs by coffee shops should be legalized.", "Both purchase and sale of soft drugs by coffee shops should not become legal."),
    ("The government should make Dutch-language education more frequently mandatory at universities and colleges.", "The government should no longer make education in Dutch compulsory at universities and colleges."),
    ("New residential areas should consist of at least 40 percent green space.", "New housing developments should not consist of at least 40 percent social housing."),
    ("There should be no new restrictions on the activities of farming businesses.", "There should be new restrictions on farm activities."),
    ("The Netherlands should accept more refugees than it currently does.", "The Netherlands should not accept more refugees than it does now."),
    ("Childcare should be free for all parents for at least three days a week.", "Child care should not become free for all parents at least three days a week."),
    ("People should always have the choice of whether to wear a face mask.", "People should not always be able to choose whether to wear a mouthpiece."),
    ("The Netherlands should exit the European Union (EU).", "The Netherlands should not leave the European Union (EU)."),
    ("Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists.", "There should not be a tax per kilometer driven for motorists instead of the tax on car ownership."),
    ("During the upcoming New Year's Eve, it should be allowed to set off decorative fireworks again.", "Next New Year's Eve it should not be allowed to set off decorative fireworks again."),
    ("Less funding should go to public broadcasting.", "There should not be less money for public broadcasting."),
    ("Instead of the existing health insurance companies, there should be a national healthcare fund for everyone.", "There should not be a national health care fund for everyone instead of the existing health insurance companies."),
    ("All entrepreneurs should pay the same health premium regardless of income.", "Not all entrepreneurs should pay the same amount of health premium regardless of income."),
    ("The independence of the judiciary from parliament and the government should be strengthened.", "The independence of the judiciary from parliament and government should not be strengthened."),
    ("The share of defense spending in Poland's GDP should be further increased.", "The share of defense spending in Poland's GDP should not be increased further."),
    ("Poland should move away from coal mining no later than 2040.", "Poland should not move away from coal mining by 2040."),
    ("Poland should have grain imports from Ukraine blocked.", "Poland should not lead to the blocking of grain imports from Ukraine."),
    ("Public media funding from the state budget should be limited.", "Funding of public media from the state budget should not be restricted."),
    ("Social transfers should be increased to reduce the effects of inflation on citizens.", "Social transfers should not be increased to limit the effects of inflation on citizens."),
    ("The powers of the secret services to track the activities of citizens on the Internet should be limited.", "The powers of the secret services to track the activities of citizens on the Internet should not be restricted."),
    ("The state should provide a free nursery place for every child.", "The state should not provide a free nursery place for every child."),
    ("Schools should have more freedom to choose the content covered in the curriculum.", "Schools should not have more freedom to choose the content covered in the curriculum."),
    ("The state should build low-rent apartments for rent.", "The state should not build low-income rental housing."),
    ("Taxes should be increased for top earners.", "Taxes should not be increased for top earners."),
    ("Early retirement should be introduced for those who have worked a certain number of years, regardless of their age.", "There should be no early retirement for those who have worked a certain number of years, regardless of their age."),
    ("Abortion laws should be liberalized.", "Abortion laws should not be liberalized."),
]

RULE_NAME = "Negation"

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
    # figA.savefig("negation_head_bars_by_index.png", dpi=200)
    # print("Saved: appendix_figA_head_bars_by_index.png")

    def polarity_from_probs(p: np.ndarray, tie_eps: float = 1e-6) -> str:
        """
        p: shape [7], digits 1..7 probs
        neg = sum(p[0:3])  # 1,2,3
        pos = sum(p[4:7])  # 5,6,7
        neutral digit=4 is ignored for polarity, but ties/very small margins -> 'amb'
        """
        neg = float(p[0] + p[1] + p[2])
        pos = float(p[4] + p[5] + p[6])
        if abs(neg - pos) <= tie_eps:
            return "amb"
        return "neg" if neg > pos else "pos"


    def count_unflips_for_single_patch(
        model,
        processor,
        pairs: List[Tuple[str, str]],
        kind: str,          # "attn" or "mlp"
        layer_idx: int,     # e.g. 23 or 22
        restoration_thresh: float = 0.0,
        tie_eps: float = 1e-6,
        verbose: bool = True,
    ):
        assert kind in ("attn", "mlp", "block")

        unflip_count = 0
        eligible_flip_count = 0  # how many actually flipped (clean pol != corrupt pol)
        details = []             # per-pair record

        for i, (b, v) in enumerate(pairs, 1):
            if verbose and (i % 10 == 1 or i == len(pairs)):
                print(f"[{kind} L{layer_idx}] pair {i}/{len(pairs)}")

            enc_clean = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(b))
            enc_corrupt = encode_for_next_token(processor, model, SYSTEM_PROMPT, build_user_prompt(v))

            logits_clean = forward_logits_only(model, enc_clean)
            logits_corrupt = forward_logits_only(model, enc_corrupt)

            clean_probs_t = digit_probs_from_logits_full(logits_clean, enc_clean, TEMP_FOR_PROBS)
            corrupt_probs_t = digit_probs_from_logits_full(logits_corrupt, enc_corrupt, TEMP_FOR_PROBS)

            clean_probs = clean_probs_t.detach().float().cpu().numpy()
            corrupt_probs = corrupt_probs_t.detach().float().cpu().numpy()

            pol_clean = polarity_from_probs(clean_probs, tie_eps=tie_eps)
            pol_corrupt = polarity_from_probs(corrupt_probs, tie_eps=tie_eps)

            # 只统计“确实 flip 了”的
            flipped = (pol_clean in ("neg", "pos")) and (pol_corrupt in ("neg", "pos")) and (pol_clean != pol_corrupt)
            if flipped:
                eligible_flip_count += 1

            clean_cache = collect_clean_cache(model, enc_clean)

            spec = {"block": [], "attn": [], "mlp": []}
            spec[kind] = [layer_idx]

            with patch_context(model, enc_corrupt, clean_cache, spec):
                logits_patched = forward_logits_only(model, enc_corrupt)
                patched_probs_t = digit_probs_from_logits_full(logits_patched, enc_corrupt, TEMP_FOR_PROBS)

            patched_probs = patched_probs_t.detach().float().cpu().numpy()
            pol_patched = polarity_from_probs(patched_probs, tie_eps=tie_eps)

            r = normalized_restoration(w_1d, clean_probs_t, corrupt_probs_t, patched_probs_t)
            restoration = float(r.item())

            # unflip 成功：patched 极性回到 clean；且原本是 flipped；且 restoration 超过门槛
            unflipped = flipped and (pol_patched == pol_clean) and (restoration > restoration_thresh)

            if unflipped:
                unflip_count += 1

            details.append({
                "idx": i,
                "clean_pol": pol_clean,
                "corrupt_pol": pol_corrupt,
                "patched_pol": pol_patched,
                "flipped": flipped,
                "unflipped": unflipped,
                "restoration": restoration,
            })

        return unflip_count, eligible_flip_count, details


    # ---- 在 main() 里加载完 model/processor 后，直接调用： ----
    # 你要的两个统计：
    attn23_unflip, attn23_flipped, attn23_details = count_unflips_for_single_patch(
        model=model,
        processor=processor,
        pairs=PAIRS,
        kind="attn",
        layer_idx=23,
        restoration_thresh=0.0,   # 想更严格可改 0.3 / 0.5
        verbose=True,
    )

    mlp22_unflip, mlp22_flipped, mlp22_details = count_unflips_for_single_patch(
        model=model,
        processor=processor,
        pairs=PAIRS,
        kind="mlp",
        layer_idx=22,
        restoration_thresh=0.0,
        verbose=True,
    )

    print("\n================ RESULTS ================")
    print(f"ATTN layer 23: unflip {attn23_unflip} / flipped-eligible {attn23_flipped} (total pairs={len(PAIRS)})")
    print(f"MLP  layer 22: unflip {mlp22_unflip} / flipped-eligible {mlp22_flipped} (total pairs={len(PAIRS)})")

    # 可选：把细节导出成 csv，方便你之后筛选看哪些句子被 unflip 了
    import csv
    with open("unflip_attn23_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(attn23_details[0].keys()))
        w.writeheader()
        w.writerows(attn23_details)

    with open("unflip_mlp22_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mlp22_details[0].keys()))
        w.writeheader()
        w.writerows(mlp22_details)

    print("Saved: unflip_attn23_details.csv, unflip_mlp22_details.csv")


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
        title=f"Activation patching\n{RULE_NAME} (flip pairs n={patch_stats['n_pairs']})",
        stats=patch_stats,
        ylabel="Normalized restoration score",
    )
    fig1.savefig("negation_patching_one_rule.png", dpi=200)
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
    # fig2.savefig("negation_ablation_one_rule.png", dpi=200)
    # print("Saved: fig2_ablation_one_rule.png")

    # # Show all three figures
    # plt.show()


if __name__ == "__main__":
    main()
