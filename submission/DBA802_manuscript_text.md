# Research manuscript — text-accessible companion

> **DBA 802 research draft, not peer-reviewed or submitted to Elsevier.** The authoritative formatted document with embedded figures/tables is the DOCX/PDF Blackboard package. This text version was extracted from that document for convenient review and searching; figures and tables may not reproduce identically in Markdown.

Outcome-Label Maturity and Knowledge-Rule Reliability in E-Commerce Fulfillment Delay

Prediction: A Temporal Audit

Madhukar Reddy Police

Corresponding author: Madhukar Reddy Police | Affiliation: Midwest University, 851 Parr Road, Wentzville, Missouri 63385,

United States | Email: madhukar.reddy246@gmail.com

Abstract

Operational decision-support models face two linked uncertainties: whether domain-derived risk rules

generalize and whether delivery outcomes are observable when validation decisions must be made. We

audit 96,456 eventually delivered orders from the Brazilian Olist dataset. Five chronological test months

(March–July 2018) compare rule-augmented predictors, held-out calibration, and a three-month-lagged,

more outcome-mature calibration window. Immediately preceding-month validation cohorts had only

47.0%–72.6% observable delivery outcomes at deployment cutoffs; observed lateness rates were

substantially below their eventual cohort rates. In a controlled within-month sensitivity experiment, older-

cohort global calibration reduced Brier error against raw XGBoost in March and July, was approximately

unchanged in April, and worsened it in May and June. Nonnegative rule corrections added no consistent

gain. These results identify selective outcome availability as a confounder in retrospective assessments of

knowledge-guided prediction. The study contributes a transparent temporal audit, not an algorithm

outperforming the baseline; further prospectively observed and externally sourced data are needed.

Keywords: delayed labels; e-commerce logistics; rule validity; knowledge-based systems; calibration;

temporal evaluation; decision support

1. Introduction

Delivery promises are operational decisions, not merely point forecasts. The Olist transaction archive [1]

enables retrospective order-level analysis; studies of retail lead-time promises and integrated delay

prediction show the value of linking forecasting to operational decisions [2,3]. Last-mile routes and service

capacity also vary by courier and fulfillment design [4–6]. A recent study has already combined e-

commerce delay prediction and cost-sensitive intervention [7], so applying machine learning to late

delivery cannot by itself establish novelty.

The use of explanations in operational AI requires caution: explainable interfaces and human-centered

decision-support designs have been discussed in the supply-chain and decision-systems literature [8–10].

Theoretical work distinguishes inherently interpretable decision models from after-the-fact explanation,

while tree-based explanation tools clarify rather than validate predictions [11,12]. Our operational "rules"

are therefore treated as unverified hypotheses, not documented carrier policies.

More fundamentally, models face shifting data environments [13–16] and delayed outcome labels [17,18].

Drift explanation, label-free diagnostics and cost-sensitive stream adaptation already constitute active

research [19–22]. Recent courier scheduling and responsible supply-chain AI work further separates

predictive scores from measured operational benefit [23,24]. Against this literature, our contribution is a

reproducible *diagnostic case study*: examine whether the ability to observe outcomes at a decision cutoff

changes the interpretation of apparently failing knowledge-rule adjustments. We do not claim the first

delayed-label method or a new generalizable algorithm.

2. Data, estimand and models

The Olist source comprises historical Brazilian marketplace orders [1]. The analytic cohort includes 96,456

records marked delivered, with available order approval, predicted delivery date, actual delivery date and at

least one order item. The binary outcome equals 1 only when the actual delivered calendar date is later than

the platform-estimated calendar date; this is lateness against an estimated date, not an independently

documented contractual SLA. Results are conditional on *eventually delivered* orders and cannot be

generalized to cancellations, unresolved shipments or all incoming orders.

Available-at-approval predictors encompass order/approval time, the promise window, item counts, seller

counts, product weights, price, freight and approximate seller–customer distance. Shipment and delivery

timestamps, final review scores and post-approval status are not used as prediction inputs. Geographical

distances are ZIP-prefix approximations, not routing or live carrier-time measurements. The main methods

include logistic regression, rules-only logistic regression, XGBoost [25] and augmented XGBoost; the last

of these adds heuristic features rather than enforcing formal knowledge constraints.

Three proposed rules represent a tight delivery commitment (derived from an assumed 500 km/day travel

heuristic), multi-seller coordination and parcel mass over 10 kg. None has been independently elicited or

verified as an Olist business policy. A previous validation-only constrained calibration experiment used a

nonnegative rule-weight model (Eq. 1):

                    q(x) = sigmoid{ a + b·logit[p(x)] + Σⱼ wⱼ·rⱼ(x) }, b ≥ 0, wⱼ ≥ 0.   (1)

 Here p(x) is the fitted base-model probability, rⱼ(x) is a predeclared binary rule, and a, b, wⱼ are calibration

 parameters. An L2-regularized validation log-loss objective was used; positivity prevents the overlay from

 reversing assumed directions but cannot establish that those directions are true. A signed, validation-only

 reliability gate was also examined separately in supplementary analysis.

 2.1 Cutoff-respecting outcome-maturity audit

 For each monthly test cutoff c from March through July 2018, define the "recent" calibration cohort as

 orders placed in [c − 1 month, c) and the "older" cohort as [c − 3 months, c − 2 months). Only orders

 delivered before c enter calibration fits. Base XGBoost is fit on information available before the older

 calibration window, and a common fitted base predictor is used to compare calibration options within a

 month. The test set is orders purchased in [c, c + 1 month); test outcomes are never consulted during

 training, parameter choice or calibration. A retrospective audit calculates the eventual delivered outcome

 rate solely to reveal the selection induced by early label availability.

 Reported comparisons are five dependent historical month-level tests, not five statistically independent

 environments. Mean squared probability error (Brier score), Brier = n⁻¹ Σᵢ (pᵢ − yᵢ)², is the primary

 calibration metric. A separate earlier experiment evaluated average precision (PR-AUC), but its training

 windows differ and its scores cannot be treated as controlled contrasts against the new audit. No inferential

 p-value is claimed from just five related months.

 3. Results

 Table 1. Availability of retrospectively ascertainable outcomes at each cutoff. “Recent observed late” uses

 only labels available at cutoff; “eventual recent late” uses hindsight and is never passed to a model. The

 older cohort is purchased three months before each cutoff.

                                        Recent observed
Test month         Recent known                            Eventual late†      Older known        Recent N
                                        late

Mar 2018           47.0%                0.45%              14.13%              99.8%              6,555

Apr 2018           50.0%                4.77%              18.96%              99.7%              7,003

May 2018           62.1%                1.16%              4.50%               99.5%              6,798

Jun 2018           65.3%                0.84%              6.56%               99.9%              6,749

Jul 2018           72.6%                0.16%              1.16%               99.9%              6,096

 †Hindsight-based outcome rate over the eventual delivered recent cohort. There is no assumption that these

 labels were available at the cutoff.

 Only 47.0%–72.6% of the recent calibration cohort was labeled at its monthly decision cutoff, compared

 with 99.5%–99.9% for the older cohort (Table 1; Figure 1). Completion-time selection mechanically favors

 faster deliveries; it does not, on its own, prove concept drift or a causal effect of knowledge rules.

 Figure 1. Recent versus older outcome-label observability by test month. Reproduced programmatically

 from the outcome-maturity audit CSV.

 Table 2. Brier error in the controlled sensitivity experiment. Models within a row use the same base

 XGBoost; lower values indicate better probability accuracy.

Month                  Raw XGBoost             Recent global          Older global           Older + rules

Mar 2018               0.16908                 0.18671                0.15638                0.15635

Apr 2018               0.04253                 0.04174                0.04255                0.04255

May 2018               0.06076                 0.06451                0.06502                0.06502

Jun 2018               0.01153                 0.01143                0.02607                0.02607

Jul 2018               0.03231                 0.03347                0.03195                0.03195

 The older, almost fully observed validation window is not universally preferable: its global calibration

 reduces Brier error against raw XGBoost in March and July, is effectively unchanged in April, and worsens

 it in May and June. Knowledge constraints do not rescue the poorly performing months (Table 2; Figure 2).

Figure 2. Change in test Brier error from raw XGBoost, by calibration strategy. Negative values indicate

improved calibration. Different-month base fits are not comparable as controlled treatment effects.

The tight-commitment indicator fires on only 27 of 96,456 eligible orders, and multiple-seller orders had

lower raw observed lateness than single-seller orders. In the preceding exploratory signed rule gate, a rule

adjustment was accepted only for May 2018; test Brier worsened from 0.06043 to 0.06129. We therefore

cannot interpret the observed failures exclusively as rule drift; label-maturity bias and rule misspecification

are credible alternative explanations.

4. Discussion, integrity and limitations

An evaluation that uses only available completed shipments is cutoff-feasible but may be unrepresentative.

Conversely, one that borrows later delivery outcomes would leak information unavailable to deployed

decision-makers. The practical choice is to report the *label-observation process* alongside performance,

and to compare a prespecified, sufficiently mature evaluation cohort with a recent cohort. This does not

eliminate temporal staleness; indeed the near-complete older cohort sometimes calibrates worse. This

distinction matters for both knowledge-engineered rules and ordinary statistical models.

The study is diagnostic rather than prospective. It uses one platform, five historically overlapping test

months, labels conditional on eventual delivery, heuristic rather than elicited business knowledge, no

costed interventions, and no prospective or external validation. Observed Brier differences are not

statistically independent estimates of causal business value. More broadly, prior work already addresses

concept drift, rule knowledge drift and delayed feedback [13–22]; the present record does not substantiate a

novel superior algorithm suitable for a strong KBS methods claim.

5. Conclusion

On Olist data, outcome-label maturity substantially changes what can be learned at a realistic order cutoff.

A more mature older calibration cohort changes probability accuracy but provides no consistent

improvement, and heuristic knowledge corrections are not reliably beneficial. These findings support an

explicit reporting protocol for label observability and rule provenance. They do not establish generalization

beyond Olist or a successful novel knowledge-based decision system; prospective, externally validated

research remains necessary.

Declarations and availability

Data and code availability: The Olist dataset is available from the original source [1]; raw transaction files are not redistributed.

Experiment code and historical outputs are available at https://github.com/Madhu2497/explainable-ai-ecommerce-fulfillment. The

public repository does not contain the journal manuscript.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the author used OpenAI ChatGPT for scientific writing assistance, bibliographic discovery,

code-review support, and documentation. After using the tool, the author reviewed and edited the content as needed and takes full

responsibility for the content of the publication.

References

[1] Olist, Brazilian E-Commerce Public Dataset by Olist, Kaggle dataset (2018).

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

[2] T. Zhang, X. Xu, Data-driven lead time forecasting and promising for an online retail platform,

International Journal of Production Economics (2026). https://doi.org/10.1016/j.ijpe.2026.109985

[3] M.A. Zaghdoudi, S. Hajri-Gabouj, F. Ghezail, S. Darmoul, C. Varnier, N. Zerhouni, Collaborative and

integrated data-driven delay prediction and supplier selection optimization: A case study in a furniture

industry, Computers & Industrial Engineering (2024). https://doi.org/10.1016/j.cie.2024.110590

[4] A. Pegado-Bardayo, A. Lorenzo-Espejo, J. Muñuzuri, L. Onieva, A predictive framework for last-mile

delivery routes considering couriers’ behavior heterogeneity, Computers & Industrial Engineering (2024).

https://doi.org/10.1016/j.cie.2024.110665

[5] B. Sawik, Optimizing Last-Mile Delivery: A Multi-Criteria Approach with Automated Smart Lockers,

Capillary Distribution and Crowdshipping, Logistics (2024). https://doi.org/10.3390/logistics8020052

[6] R. Srinivasan, J. Szmerekovsky, Exploring the Impact of Delivery Robots on Last-Mile Delivery

Capacity Planning Using Simulation, Logistics (2025). https://doi.org/10.3390/logistics9040156

[7] Y. Zhang, Y. Wang, Y. Wang, A Deep Tabular Learning Framework for E-Commerce Delivery Delay

Prediction and Cost-Sensitive Operational Risk Mitigation, Algorithms (2026).

https://doi.org/10.3390/a19070597

[8] K. Sadeghi R., D. Ojha, P. Kaur, R.V. Mahto, A. Dhir, Explainable artificial intelligence and agile

decision-making in supply chain cyber resilience, Decision Support Systems (2024).

https://doi.org/10.1016/j.dss.2024.114194

[9] K. Coussement, M.Z. Abedin, M. Kraus, S. Maldonado, K. Topuz, Explainable AI for enhanced

decision-making, Decision Support Systems (2024). https://doi.org/10.1016/j.dss.2024.114276

[10] V.C. Storey, A.R. Hevner, V.Y. Yoon, The design of human-artificial intelligence systems in decision

sciences: A look back and directions forward, Decision Support Systems (2024).

https://doi.org/10.1016/j.dss.2024.114230

[11] C. Rudin, Stop explaining black box machine learning models for high stakes decisions and use

interpretable models instead, Nature Machine Intelligence (2019). https://doi.org/10.1038/s42256-019-

0048-x

[12] S.M. Lundberg et al., From local explanations to global understanding with explainable AI for trees,

Nature Machine Intelligence (2020). https://doi.org/10.1038/s42256-019-0138-9

[13] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, G. Zhang, Learning under Concept Drift: A Review, IEEE

Transactions on Knowledge and Data Engineering (2019). https://doi.org/10.1109/TKDE.2018.2876857

[14] Y. Sun, J. Mi, C. Jin, Entropy-based concept drift detection in information systems, Knowledge-Based

Systems (2024). https://doi.org/10.1016/j.knosys.2024.111596

[15] G.J. Aguiar, A. Cano, A comprehensive analysis of concept drift locality in data streams, Knowledge-

Based Systems (2024). https://doi.org/10.1016/j.knosys.2024.111535

[16] Y. Lin, Y. Duan, C. Jin, F. Li, Research on knowledge drift based on interaction matching,

Knowledge-Based Systems (2025). https://doi.org/10.1016/j.knosys.2025.113664

[17] J. Komorniczak, P. Ksieniewicz, P. Zyblewski, Structuring the processing frameworks for data stream

evaluation and application, Pattern Recognition (2026). https://doi.org/10.1016/j.patcog.2025.112516

[18] M. Botacin, H. Gomes, Towards more realistic evaluations: The impact of label delays in malware

detection pipelines, Computers & Security (2025). https://doi.org/10.1016/j.cose.2024.104122

[19] L. Yang, J. Cheng, Y. Luo, T. Zhou, X. Zhang, Detecting and rationalizing concept drift: A feature-

level approach for understanding cause–effect relationships in dynamic environments, Expert Systems with

Applications (2025). https://doi.org/10.1016/j.eswa.2024.125365

[20] M. Roostaee, M.H. Soheilian, ReDiST: A lightweight residual analysis framework for fast and

accurate unsupervised drift detection in intelligent systems, Knowledge-Based Systems (2026).

https://doi.org/10.1016/j.knosys.2025.115247

[21] A. Diaz Iglesias et al., Robust concept drift handling in dynamic industrial systems: A multi-objective

optimization approach, Knowledge-Based Systems (2026). https://doi.org/10.1016/j.knosys.2025.115041

[22] Y. Chen, X. Yang, H.-L. Dai, Cost-sensitive continuous ensemble kernel learning for imbalanced data

streams with concept drift, Knowledge-Based Systems (2024).

https://doi.org/10.1016/j.knosys.2023.111272

[23] Z. Saleh, A. Al Hanbali, A. Baubaid, Enhancing courier scheduling in crowdsourced last-mile delivery

through dynamic shift extensions: A deep reinforcement learning approach, Computers & Industrial

Engineering (2026). https://doi.org/10.1016/j.cie.2025.111693

[24] L. Li, R. Shi, R. Zhang, L. Chen, How can firms leverage responsible artificial intelligence to build

competitive advantages? The role of supply chain justice and complexity, International Journal of

Production Economics (2026). https://doi.org/10.1016/j.ijpe.2026.110010

[25] T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System, Proceedings of the 22nd ACM

SIGKDD Conference (2016). https://doi.org/10.1145/2939672.2939785
