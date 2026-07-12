# Model Card: Explainability (SHAP)

## Purpose
Provides human-readable explanations for individual fraud predictions
from the Stage 1 XGBoost classifier, and global feature importance for
model governance/audit — matching real-world practice where SHAP
explanations are shown alongside fraud scores in analyst review
dashboards (compliance/regulatory requirement in many jurisdictions,
e.g. EU AI Act treatment of fraud detection as high-risk).

## Approach
shap.TreeExplainer on the trained XGBoost model (fast, exact for
tree-based models, unlike generic model-agnostic SHAP explainers).

## Outputs
1. Global summary plot (beeswarm) — feature importance and directional
   effect across all scored transactions, for model governance.
2. Per-transaction force plot — visual breakdown of exactly which
   features pushed one specific prediction toward/away from fraud.
3. Auto-generated narrative — top-3 SHAP contributions per transaction
   turned into a plain-English sentence, e.g.: "This transaction was
   flagged because: amount_last_7_days increased fraud risk; amount
   increased fraud risk; country_changed_False decreased fraud risk."

## Validation finding
SHAP independently confirmed a limitation already documented in the
rule-based fraud-type classifier's model card: the first transaction in
a structuring sequence shows amount == amount_last_7_days (no rolling
history yet), and SHAP correctly identifies these as the dominant
drivers — the same ambiguity discovered through manual debugging in
Milestone 3 is visible directly in the model's own explanation, from
an independent method.