# DBA 802 — Blackboard submission guide

**Research project:** Outcome-Label Maturity and Knowledge-Rule Reliability in E-Commerce Fulfillment Delay Prediction: A Temporal Audit  
**Author:** Madhukar Reddy Police  
**Affiliation:** Midwest University  
**Course:** DBA 802 — Data Analytics and Strategic Decision Intelligence  
**GitHub:** https://github.com/Madhu2497/explainable-ai-ecommerce-fulfillment

## Submit these files through Blackboard

The final PDF, editable Word file, exact programmatically generated PNG figures, graphical abstract, and the supplementary code/results ZIP are distributed as the separate **DBA802 Blackboard Course Package** provided to the author in this conversation. They are **not yet present as binary files in this GitHub repository**; do not mistake the Markdown/ SVG research companions for copies of the final submission artifacts.

1. `01_DBA802_Research_Paper.pdf` — main assignment.
2. `02_DBA802_Research_Paper_Editable.docx` — upload if Word is required.
3. `03_Figure_1_Label_Maturity.png` and `04_Figure_2_Calibration_Sensitivity.png` — two programmatically produced research figures; the manuscript already embeds two figures.
4. `05_Graphical_Abstract.png` — if requested.
5. `07_Supplementary_Code_and_Results.zip` — experiments, supplementary results, and reproducibility evidence.

**Two tables are already inside the manuscript.** The primary repository code/results files are also browseable in the parent directory.

**Journal submission proof:** Not yet available. Do not describe the research as submitted to Elsevier or attach fabricated confirmation. If the course rubric requires proof, an actual journal-portal confirmation must be obtained separately.

## Public repository contents

- `experiment.py`: initial comparison and chronological five-month validation.
- `decision_validation.py`: fixed operational-rule constrained probability-layer sensitivity analysis.
- `rule_reliability.py`: conservative rule-gating experiment.
- `label_maturity_sensitivity.py`: delayed-outcome visibility and calibration-cohort sensitivity audit.
- `results/`: CSV results and original baseline figures.
- `submission/DBA802_manuscript_text.md`: text-accessible version of the current manuscript (not the authoritative formatted PDF/Word).
- `submission/Figure_1_Label_Maturity.svg` and `submission/Figure_2_Calibration_Sensitivity.svg`: vector reconstructions based on the validated output CSVs; for the **exact manuscript figures**, use the PNG files in the local Blackboard package.

## Reproduce key validation outputs

Download the Olist Brazilian e-commerce dataset from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and keep the original CSV files inside `archive.zip` in the project directory. The repository intentionally does **not** distribute the original Olist dataset.

```bash
python -m pip install -r requirements.txt
python experiment.py --data archive.zip --out results
python experiment.py --data archive.zip --out results --rolling
python decision_validation.py --data archive.zip --out results
python rule_reliability.py --data archive.zip --out results
python label_maturity_sensitivity.py --data archive.zip --out results
```

The historical Olist test window comprises five monthly held-out cohorts during March–July 2018. The evaluation population consists only of eventually delivered orders with observable outcome dates, so estimates do not apply directly to undelivered/cancelled orders. The immediately preceding month's calibration cohort included just 47.0%–72.6% known outcomes by the test cutoff; a three-month-lagged cohort had over 99% known outcomes but suffered temporal staleness. These are **retrospective sensitivity analyses**, not prospective field trials.

## Scientific integrity and status

- No demonstrated predictive advantage of the heuristically knowledge-guided models over ordinary XGBoost across the five monthly holdouts.
- Stronger novel knowledge-based method, independent external replication, and remaining bibliographic verification are needed before claiming journal readiness.
- Independent prospective validation and treatment-effect/cost evaluation have **not** occurred.
- The manuscript is a research draft prepared for grading, **not** an accepted or submitted journal paper.
