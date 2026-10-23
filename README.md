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