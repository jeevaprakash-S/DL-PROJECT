# Network World Model IDS — Version 2

This project predicts (1) the next scaled network-flow state and (2) the likely stage of the next flow from a sequence of prior flows. It is a **multi-task Transformer World Model**, not merely a per-flow IDS.

Version 2 corrects the failed V1 training setup without promising an untested score. V1 recorded macro-F1 ≈0.22 and near-zero F1 for Delivery, Exploitation, C2, and ActionsOnObjectives. The main causes were extreme imbalance (only 433 Exploitation training windows), a capped class-weight loss, scaling fitted with future flows, and sequences that could mix unrelated hosts.

## What changed

- Windows never span CSV/source-file or split boundaries. Per-Destination-IP grouping is supported but disabled by default: enable it only when the label audit confirms each stage has validation/test examples.
- Train/validation/test partitions occur before imputation and scaling. The median imputer and `StandardScaler` fit only the training flows.
- `label_audit.json` records every raw and unmapped label. Unknown labels are excluded rather than silently treated as an attack stage. Review it and explicitly amend `LABEL_TO_STAGE_RULES` in `src/config.py` if needed.
- Class-aware sampling uses inverse-square-root frequency, which improves exposure of rare stages while avoiding excessive duplication of the rarest windows.
- The classifier uses class-balanced focal loss (effective-number weighting, capped only after normalization) and saves the best model by validation macro-F1 rather than loss.
- A compact 64-wide, 2-layer Transformer and a bounded number of sampled updates make CPU/Mac runs practical. Apple Silicon MPS is used automatically when available.
- Inference uses both predicted non-benign probability and a p95 benign-validation next-state forecast error. `explain.py` creates attention and gradient×input feature-attribution artifacts.
- All six stages remain training and inference outputs. When a strict chronological test split has fewer than 100 independent windows for a stage, `evaluation_scope.json` flags it as experimental and reports a separate primary macro-F1 over the sufficiently supported stages. This prevents a large benign class or a handful of rare test windows from being presented as a dependable six-stage result.

## Install and data location

Use Python 3.10–3.12, then from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
```

Place CICIDS2017/2018 CSV files in `data/raw/`. Do not mix arbitrary CSV schemas in the same run.

## Run sequence

First inspect exact labels; this does not build sequences:

```bash
python src/data_preprocessing.py --inspect
cat outputs/label_audit.json
```

If any label is unmapped, update the explicit mapping in `src/config.py` and run the audit again. The stage mapping is a project-defined analytical taxonomy, not a claim that CICIDS natively supplies kill-chain labels.

Build leakage-safe data arrays:

```bash
python src/data_preprocessing.py
```

Start with a short CPU sanity training run:

```bash
python src/train.py --sanity --epochs 5
```

Inspect `outputs/test_classification_report.txt`, especially every class's support and F1. Only then run a larger experiment:

```bash
python src/train.py --epochs 12 --train-steps 2500
python src/inference.py --limit 200
python src/explain.py --index 0
```

## Outputs

- `outputs/label_audit.json` — safe label-map audit and per-split coverage.
- `checkpoints/world_model_v2_best.pt` — model selected by validation macro-F1.
- `outputs/test_classification_report.txt` — held-out per-class results and confusion matrix.
- `outputs/evaluation_scope.json` — the supported headline evaluation stages, raw test support, and stages that must be described as experimental.
- `outputs/alert_threshold.json` — p95 benign validation forecast-error threshold.
- `outputs/alerts_log.json` — combined classification/novelty warning events.
- `outputs/attention_seq*.png`, `feature_attribution_seq*.png`, and `explanation_seq*.json` — explanation artifacts.

## Interpretation and limitations

Balanced training cannot invent diversity for a class with very few distinct flows. A non-zero rare-class F1 in a small sanity run is a pipeline check, not a performance guarantee. Keep the held-out test set naturally imbalanced, report macro-F1 plus per-class support, and avoid changing mappings after examining test outcomes. The split is chronological within each destination/source stream; it is appropriate for forecasting, while cross-day/domain generalization needs a separately designed evaluation.
