# Industrial Asset Reliability Platform

A predictive maintenance platform for rotating industrial equipment - built as a personal portfolio project inspired by realistic oil and gas maintenance challenges.

## Overview

Three complementary AI capabilities, sharing a common data foundation and MLOps infrastructure.

| Module | Task | Status |
|---|---|---|
| AssetPulse | Failure classification | Complete |
| RULSense | Remaining Useful Life regression | Complete |
| AssetGuardian | Anomaly detection and root cause analysis | Complete |

## AssetPulse

Predicts whether equipment will fail within 7, 14, or 30 days.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.999 | 0.708 | 0.829 | 0.947 |
| XGBoost | 0.947 | 0.685 | 0.795 | 0.940 |
| LightGBM | 0.916 | 0.682 | 0.782 | 0.939 |

Stack: scikit-learn, XGBoost, LightGBM, SHAP, MLflow, Azure ML Batch Endpoints, Azure DevOps

## RULSense

Estimates hours remaining until next failure using 60-hour sensor sequences.

| Model | RMSE hours | MAE hours | MAPE |
|---|---|---|---|
| GRU | 642 | 506 | 92.9% |
| LSTM | 698 | 561 | 110.3% |

Stack: PyTorch LSTM and GRU, MLflow tracking, sliding window sequences

## AssetGuardian

Unsupervised anomaly detection with rule-based root cause diagnosis.

| Metric | Value |
|---|---|
| Precision | 0.356 |
| Recall | 0.834 |
| F1 | 0.499 |

Root causes diagnosed: bearing wear, lubrication degradation, flow restriction, motor winding failure.

Stack: Isolation Forest, z-score feature attribution, engineering rules

## Key Design Decisions

Asset-level train test splits everywhere to prevent data leakage.

Labels derived from historical failure events, not synthetic shortcuts.

Batch scoring chosen over real-time API since maintenance planning is a daily or hourly cadence decision.

## Known Limitations

Synthetic data simplifies real-world sensor noise.

Drift detection is sensitive to distribution shape at large sample sizes.

RULSense occasionally predicts negative RUL near failure points, handled via clipping.

AssetGuardian precision reflects the expected tradeoff of unsupervised anomaly detection.


## Production Optimization Copilot

Linear Programming-based production allocation across 15 wells sharing pipeline and compressor capacity, integrated directly with AssetPulse/RULSense health scores.

| Scenario | Total Production | Impact |
|---|---|---|
| Baseline | 4499.9 bpd | - |
| Pipeline +500 bpd | 4999.9 bpd | +11.1% |
| WELL-009 degrades 50% | 4499.9 bpd | 0% (pipeline remains bottleneck) |
| Compressor +300 bpd | 4499.9 bpd | 0% (not the binding constraint) |

Stack: PuLP (CBC solver), Linear Programming, sensitivity analysis via what-if simulation

Key design decision: A fairness constraint (minimum 30% capacity per well) was added after discovering the unconstrained solver would completely zero out healthy wells if that reached the same total output faster - a mathematically optimal but operationally unrealistic result. This demonstrates a core optimization lesson: solvers only respect what is explicitly encoded as a constraint.


## Intelligent Maintenance Recommendation Engine

Given a diagnosed failure mode (from AssetGuardian), recommends specific maintenance actions - parts, inspections, tools, follow-up work orders - using two complementary approaches.

| Approach | Precision | Notes |
|---|---|---|
| Cosine Similarity (Phase 1) | 100% (by construction) | Cannot generalize to novel failure modes |
| Two-Tower Embeddings (Phase 2) | 65% (Precision@5) | Genuine generalization, trades precision for coverage |

Cold-start handling: novel failure modes with zero historical data fall back to category-based similarity (mechanical/hydraulic/electrical), explicitly flagged as lower confidence requiring engineer verification.

Stack: scikit-learn (cosine similarity), PyTorch (two-tower embedding model), negative sampling, Precision@K evaluation

Full integration: AssetGuardian's root cause diagnosis feeds directly into this engine in the dashboard - each high-risk asset shows its diagnosed failure mode alongside specific recommended actions in one view.

## Reproducing This Project

pip install -r requirements.txt

python src/generate_dataset.py

python src/validation/schema.py

python src/features/engineering.py

python src/classification/train_assetpulse_mlflow.py

python src/serving/batch_scoring.py

python src/rul/train_rulsense_mlflow.py

python src/rul/batch_scoring_rul.py

python src/anomaly/train_assetguardian.py

python src/anomaly/root_cause_analysis.py

python src/monitoring/drift_detection.py
