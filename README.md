# Explainable Knowledge-Based AI for E-Commerce Fulfillment Delays

DBA 802 research prototype | **real Olist transactions; preliminary results, not a submitted/accepted journal paper.**

## Research question

Can transparent business feasibility rules improve **at-order-approval** prediction of whether an eventually delivered order arrives **after its promised calendar date**?

## Obtain and run

1. Download the *Brazilian E-Commerce Public Dataset by Olist* from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce.
2. Keep the original CSVs in a ZIP file named `archive.zip`, containing at least orders, order items, products, sellers, customers and geolocation files. The original dataset is **not included or redistributed** here.
3. Install requirements and execute:

```bash
python -m pip install -r requirements.txt
python experiment.py --data archive.zip --out results
# Additional strictly chronological monthly holdouts + paired bootstrap intervals:
python experiment.py --data archive.zip --out results --rolling
```

The script generates **exactly two experimental figures** (`results/figure1_precision_recall.png`, `results/figure2_calibration.png`) and **two experimental tables** (`results/table1_model_results.csv`, `results/table2_knowledge_ablation.csv`). `results/sample_audit.csv` records the cohort and temporal splits. The optional `--rolling` run adds `results/rolling_validation.csv` with five out-of-time monthly evaluations and paired, exploratory order-level bootstrap intervals (500 draws per comparison). It creates no additional main figures or main tables.

## Design and safeguards

- **Population:** Orders marked `delivered` with observable approval, promised date, delivered date and at least one order item. Results do not generalize to undelivered/cancelled orders.
- **Target:** `1` if actual delivered **calendar date** is later than the Olist *estimated delivery date*, otherwise `0`. This is late relative to a platform estimate, not a documented contractual service-level agreement.
- **Prediction moment:** Approval. Features include purchase/approval timing, promised delivery window, item and seller information, product size/weight, freight and price, geographic seller-to-customer distance, and state match. **No actual shipping, actual delivery, final status, review score, or shipping-limit timestamp is used as a predictive feature.**
- **Spatial aggregation:** ZIP-prefix median coordinates; seller locations for multi-seller orders are averaged. Distances are approximate great-circle distances and are not driving routes.
- **Rule signals:** A heuristic buffer `promised_window - [1 + distance_km/500 + multi_seller_indicator]`, tight commitment, multi-seller coordination, and heavy-package flag. The 500 km/day, 1 day, and 10 kg values are **unvalidated assumptions**, not Olist operational policies. Rule efficacy is tested by ablation, not assumed.
- **Chronological holdout:** Earliest 70% development, next 15% validation, latest 15% held-out test by purchase timestamp. Development orders with delivery outcomes after the following boundary are removed (purged); validation orders with outcomes after the test boundary are removed. Preprocessing is fitted on development only. Decision thresholds maximize validation F1; final results are computed once on the held-out test set.
- **Models:** Prior-only baseline, logistic regression, rules-only logistic regression, XGBoost, and XGBoost + knowledge-rule features. Same split, fixed seed 2026. Note that injecting rule signals into XGBoost is **feature augmentation** rather than a formal knowledge-constraint optimizer.
- **Reported metrics:** PR-AUC (primary for rare late deliveries), ROC-AUC, Brier score, F1, precision, recall, alert rate, top-10%-alert recall. Threshold-dependent metrics are sensitive to the capacity of operational follow-up teams.

## Initial executable findings (single temporal split)

| Model | Test PR-AUC | Test ROC-AUC | Brier (lower is better) |
|---|---:|---:|---:|
| Logistic | 0.0765 | 0.6316 | 0.0448 |
| Rules only | **0.1294** | **0.7517** | 0.0417 |
| XGBoost | 0.0915 | 0.7107 | **0.0402** |
| XGBoost + knowledge rule signals | 0.0871 | 0.7008 | 0.0404 |

**Interpretation (single test):** Rule-only ranking was higher in this particular period; feature-augmented XGBoost did **not** improve over ordinary XGBoost. The rule-only result does not hold under the additional chronological checks below. No superior hybrid algorithm is demonstrated.

## Five additional forward-chaining tests (March-July 2018)

Each month is evaluated as a new held-out purchasing month. Training includes only orders purchased **and delivered** before the start of the preceding validation month. The immediately preceding month provides a validation set of orders with labels known by the test month. No test-month outcome is used in training, preprocessing or threshold selection. Model settings are held constant across folds.

| Test month | Late-delivery prevalence | Logistic PR-AUC | Rule-only PR-AUC | XGBoost PR-AUC | Rules + XGBoost PR-AUC |
|---|---:|---:|---:|---:|---:|
| March 2018 | 18.96% | 0.3102 | 0.1703 | 0.2517 | 0.2476 |
| April 2018 | 4.50% | 0.1592 | 0.0546 | 0.1657 | 0.1519 |
| May 2018 | 6.56% | 0.1479 | 0.0840 | 0.1298 | 0.1246 |
| June 2018 | 1.16% | 0.0846 | 0.0182 | 0.1212 | 0.1140 |
| July 2018 | 3.38% | 0.0869 | 0.0760 | 0.1103 | 0.0885 |

The rule-only approach underperforms ordinary XGBoost in **all five** months on the primary ranking metric, as does the feature-augmented XGBoost. The observed prevalence is highly variable, and PR-AUC must be interpreted against the prevalence of each held-out month. A separate exploratory paired bootstrap (500 order-level resamples per month and comparison) reports within-month differences and uncertainty bands in the CSV; order-level intervals do **not** control for repeat customers, cross-month dependence, or multiple testing. No causal interpretation is warranted. The originally encouraging single-split rule result was not robust.

**Consequence for the manuscript:** The existing rule formulation is a heuristic feature transform, *not* demonstrated knowledge-constrained reasoning. The central defensible finding is that naive business-rule augmentation may fail under temporal distribution shift; do not present this as an improved predictive method. Literature and methodological novelty review is still required before submission.

## Research integrity and next work

This repository is **not submission-ready**. Do not claim a novel superior hybrid algorithm, causal effects, generality across countries, or peer-reviewed validation. Before journal submission: pre-register the subsequent study design, validate operational decision costs, test the assumed travel speed and parcel-mass thresholds, account for selection among delivered orders, examine stronger knowledge-integration methods, add external replication, and check related work and *Knowledge-Based Systems* Short Communication guidelines. The research manuscript and references are separate work products.

### Attribution

Olist Brazilian E-Commerce Public Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. Cite and comply with the dataset's listed license and attribution requirements. Dataset files intentionally not committed.


## Step 4 — Knowledge-constrained decision experiment (exploratory, 23 October 2026)

The `--rolling` branch above studies *unconstrained feature augmentation*. A separate
`decision_validation.py` tests a real **knowledge-constrained probability model**:


`q = sigmoid(a + b * logit(p_XGB) + w_tight * tight + w_multi * multi_seller + w_heavy * heavy)`,

where `b >= 0` and each knowledge-rule coefficient `w >= 0`. The nonnegative signs are **assumptions** encoding the hypothesis that each rule signals elevated risk, NOT empirically established Olist policies. A nonnegative coefficient can be zero. The coefficients are optimized for validation-only log loss with fixed quadratic regularization (0.02), and the resulting method is compared with raw XGBoost and an otherwise identical calibration without rules. The original model is never refit using a test-month outcome. Alert allocation is evaluated at a fixed top-10% investigation capacity; no treatment/outcome data exist to estimate actual intervention effectiveness.

```bash
python decision_validation.py --data archive.zip --out results
```

This writes two **supplementary CSVs**, not extra main-paper figure/table slots:
`results/decision_validation.csv` (20 model-by-month rows) and
`results/rule_coverage_audit.csv` (descriptive full-cohort rule counts).
The code also provides exploratory paired order bootstrap intervals for **Brier-score differences** and **top-10%-alert recall differences** versus raw XGBoost. These intervals do **not** correct for repeated buyers, clustered order behavior, multiple comparisons, or temporal correlation.

### Key held-out findings (March–July 2018)

| Month | Raw XGB PR-AUC | Constrained PR-AUC | Raw Brier | Constrained Brier | Top-10% delayed-order recall, raw / constrained |
|---|---:|---:|---:|---:|---:|
| Mar | 0.2517 | 0.2516 | 0.16933 | 0.18634 | 0.1544 / 0.1544 |
| Apr | 0.1657 | 0.1657 | 0.04106 | 0.04113 | 0.3529 / 0.3529 |
| May | 0.1298 | 0.1298 | 0.06043 | 0.06414 | 0.1896 / 0.1874 |
| Jun | 0.1212 | 0.1212 | 0.01148 | 0.01143 | 0.4225 / 0.4225 |
| Jul | 0.1103 | 0.1103 | 0.03221 | 0.03346 | 0.2644 / 0.2644 |

**Conclusion:** With fixed rule signs and conservative shrinkage, nearly all constrained rule weights shrink to zero, so ranking performance is almost identical to the base model. Calibration is usually worse than the raw predictor. The expert assumptions are therefore **not supported as beneficial by this test**. A knowledge-constrained layer was genuinely implemented, but **no improved prediction or business benefit has been established**. The earlier claims of methodological superiority must not be made.

### Explicit failure analysis

- **Rare high-risk threshold:** 27 of 96,456 (0.028%) eligible orders satisfy the tight-commitment threshold, giving the rule very low coverage.
- **Contradicted coordination prior:** Multi-seller orders have an observed 1.02% late-delivery rate (1,275 orders), versus 6.85% for other orders. This association is *descriptive* and not a causal effect, but directly cautions against assuming multi-seller coordination always raises delay risk.
- **Heavy-weight prior:** Parcels with max item weight >10 kg have a 9.52% late-delivery rate versus 6.64% for others, yet the full model generally has insufficient evidence for a large additional strictly nonnegative coefficient after accounting for the other features.
- **Calibration under changing prevalence:** A preceding-month calibration layer is especially vulnerable when the next month's late-delivery prevalence changes sharply. Raw XGBoost can outperform both global and knowledge-constrained validation calibrators.
- **Interpretation:** The fixed operational rules were invented as transparent hypotheses, not drawn from documented carrier service-level agreements. We should conduct a documented literature review and seek operational subject-matter knowledge before treating them as hard constraints.

### Manuscript implications

The strongest defensible result is a **negative replication / rule-validity stress test**. Do not submit a paper claiming the algorithm outperforms XGBoost. A KBS submission would require a substantially stronger and demonstrably distinct knowledge-engineering contribution, methodological and external validation, and an updated novelty review. The class assignment's final manuscript should have **exactly two principal figures and two principal experimental tables**; the CSV outputs and rule audit may be supplementary materials.


## Step 6 — Evidence-gated rule-validity stress test (exploratory)

\`rule_reliability.py\` evaluates whether business-rule residual adjustments should be admitted or discarded **using only preceding-month, outcome-available validation records**. An internal chronological 60/40 validation split estimates signed, regularized rule effects on the earlier part and conditionally activates the overlay only if its Brier score improves on the later part. Rules also require a minimum of 30 firings and 5 outcomes in each class to qualify. No test-month outcome is used for fitting or activation.

Run \`python rule_reliability.py --data archive.zip --out results\`. Supplementary outputs: \`results/rule_reliability.csv\` and \`results/rule_gate_audit.csv\`; see \`STAGE6_AUDIT.md\` for details, data limitations, and a non-advocacy interpretation.

**Observed March–July 2018 result:** the gate disabled rule overlays in 4/5 months; in its only accepted month (May), out-of-time Brier increased from 0.060425 to 0.061294. This is **not an improvement claim**. It strengthens the documented failure/robustness analysis; the proposal remains exploratory and is not ready for journal submission. Main-paper output remains restricted to two figures and two tables; added CSVs are supplementary.

