# Stage 6 — Temporally gated knowledge-rule validation (exploratory)

**Study**: Explainable Knowledge-Based AI for Predicting E-Commerce Fulfillment Delays (Olist Brazil, 2016–2018)

## Research question
Can a conservative, pre-test empirical validity gate prevent unsupported expert-hypothesis rules from changing fulfillment-delay probabilities under temporal distribution shift?

## Setup
- The **identical XGBoost predictor, features, hyperparameters, training eligibility filters and five months** used in the prior experiments are retained.
- For each test month in March–July 2018, earlier data train the XGBoost baseline. The preceding month supplies records whose delivery outcomes are already available by the test-month start.
- The preceding month's available validation records are sorted by purchase time, then split **60%** for fitting signed, penalized residual rule effects and **40%** for deciding whether to enable the rule overlay.
- For a rule to enter fitting, at least **30** flagged orders, **5** late cases and **5** on-time cases must occur in the earlier 60% validation subset. These are exploratory prespecified pragmatic thresholds, not domain-validated operational constraints.
- A Brier score improvement exceeding an absolute **0.0001** on the latter 40% is required to activate the overlay, and at least one rule must pass support. Otherwise **fall back to untouched XGBoost**.
- Test-month labels are exclusively used for evaluation, never to select weights, thresholds, features or activation. No test-time outcome leakage is introduced.

## Results

| Test month | Supported rule | Gate activation | Raw Brier | Gated Brier | Raw AP | Gated AP |
|:--|:--|:--:|--:|--:|--:|--:|
| March 2018 | None | No | 0.169327 | 0.169327 | 0.251654 | 0.251654 |
| April 2018 | Heavy package | No | 0.041065 | 0.041065 | 0.165692 | 0.165692 |
| May 2018 | Heavy package | Yes | 0.060425 | 0.061294 | 0.129816 | 0.129826 |
| June 2018 | None | No | 0.011480 | 0.011480 | 0.121232 | 0.121232 |
| July 2018 | None | No | 0.032212 | 0.032212 | 0.110295 | 0.110295 |

Results are rounded from `results/rule_reliability.csv`; use this CSV as the numerical authority. The rule audit is at `results/rule_gate_audit.csv`.

## Interpretation and limitations
The gate rejected rule overlays in **4/5 months**. In the only accepted month, **May 2018**, the overlay improved Brier score within the preceding-month gate sample but **worsened** Brier score on the subsequent month (**0.060425 to 0.061294**). There is therefore **no evidence of an out-of-time performance gain** from the proposed gate. Improvements in the smaller, earlier gate subset can fail to persist into the future under changing late-delivery prevalence. Even when a candidate overlay is disabled, learned intercept shifts from an earlier month may differ from future prevalence; the method always falls back to the original model in those months.

The study is retrospective and **conditional on delivered orders**. It cannot predict outcomes for cancellations or demonstrate an intervention effect. The knowledge predicates (tight commitment, multi-seller coordination, heavy package) are **hypotheses**, not validated logistics service-level rules. As rule counts are often sparse, no claim of domain-level generality, methodological superiority, statistical significance across five months, or journal acceptance is justified. The 60/40 gate itself represents an exploratory design choice rather than an externally verified optimal configuration.

**Recommended next research decision**: Do not rebrand this as a successful novel method. Before further algorithmic complexity, source independently documented logistics rules, formalize a pre-registered hypothesis, and seek data with multiple independent shipping contexts or more recent periods. An honest Short Communication could report an evidence-controlled stress test if the literature review establishes a meaningful methodological contribution, but that claim remains unproven.

## Reproducibility
Run alongside repository `experiment.py` and user-obtained Olist `archive.zip`:

```bash
python rule_reliability.py --data archive.zip --out results
```

Creates only two supplementary CSVs; the manuscript still has exactly two **main figures** and two **main experimental tables**. Original Olist data are not redistributed.