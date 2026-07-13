cat > README.md << 'EOF'
# Industrial Asset Reliability Platform

A predictive maintenance platform for rotating industrial equipment (pumps, compressors, turbines, motors, ESPs) — built as a personal portfolio project inspired by realistic oil & gas maintenance challenges.

## Overview

This platform demonstrates three complementary AI capabilities for equipment reliability:

- **AssetPulse** — Failure classification (will this asset fail in the next 7/14/30 days?) ✅ *Complete*
- **RULSense** — Remaining Useful Life regression (how many hours until failure?) 🔜 *In progress*
- **AssetGuardian** — Anomaly detection + root cause analysis 🔜 *Planned*

All three share a common data foundation, feature engineering pipeline, and MLOps infrastructure — reflecting how a real enterprise predictive maintenance program is architected, rather than three disconnected models.

## AssetPulse — Equipment Failure Classification

### Business Problem
Predict whether rotating equipment will fail within 7, 14, or 30 days, using historical sensor data and maintenance records, enabling proactive maintenance scheduling instead of reactive repair or wasteful fixed-schedule replacement.

### Architecture

Synthetic SCADA-style Data Generation (12 assets, 5 equipment types)
│
Data Validation (Pandera)
│
Feature Engineering (73 features:
rolling stats, trend/lag, FFT, maintenance interval)
│
Failure Label Derivation
(looked backward from historical failure events)
│
┌──────────┼──────────┐
▼          ▼          ▼
Random    XGBoost    LightGBM
Forest
│          │          │
└──────────┼──────────┘
▼
SHAP Explainability
│
MLflow Tracking + Model Registry
│
Azure ML Batch Scoring Pipeline
│
Drift Monitoring (KS test)
│
Azure DevOps CI/CD

### Key Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** (production) | 0.999 | 0.708 | 0.829 | 0.947 |
| XGBoost | 0.947 | 0.685 | 0.795 | 0.940 |
| LightGBM | 0.916 | 0.682 | 0.782 | 0.939 |

Class imbalance (6.95% positive rate for 7-day horizon) handled via `class_weight='balanced'` (Random Forest, LightGBM) and `scale_pos_weight` (XGBoost).

### Tech Stack
Python, Pandera, scikit-learn, XGBoost, LightGBM, SHAP, MLflow, Azure ML (Batch Endpoints), Azure DevOps

### Key Design Decisions

**Asset-level train/test split, not row-level.** Splitting randomly by row would leak near-duplicate readings from the same failure event into both train and test sets. Splitting by `asset_id` ensures test assets are genuinely unseen.

**Batch scoring over real-time API.** Maintenance planning is a daily/hourly cadence decision, not a millisecond-latency use case — Azure ML Batch Endpoints match the actual business pattern.

**Labels derived from historical failure events, not synthetic shortcuts.** Even though the synthetic data generator has an internal "ground truth" operating-mode label, AssetPulse's training labels are computed by looking backward from `failure_events.csv`, mirroring exactly how a real predictive maintenance team builds training data from repair logs.

### Known Limitations
- Synthetic data encodes realistic *directional* physics (vibration rises with wear, motor current rises with resistance) but simplifies real-world sensor noise and messiness.
- Drift detection (KS test) is sensitive to distribution shape at large sample sizes — a production version would combine statistical significance with a practical magnitude threshold to avoid false-alarm drift flags on trivial shifts.

### Project Structure

asset-reliability-platform/
├── configs/config.yaml
├── data/{raw,validated,processed}/
├── src/
│   ├── generate_dataset.py
│   ├── validation/schema.py
│   ├── features/engineering.py
│   ├── classification/
│   │   ├── train_assetpulse.py
│   │   └── train_assetpulse_mlflow.py
│   ├── explainability/shap_analysis.py
│   ├── monitoring/drift_detection.py
│   ├── serving/batch_scoring.py
│   └── utils/logger.py
├── mlflow/
├── azure-pipelines.yml
└── requirements.txt

### Reproducing This Project
```bash
pip install -r requirements.txt
python src/generate_dataset.py
python src/validation/schema.py
python src/features/engineering.py
python src/classification/train_assetpulse_mlflow.py
python src/serving/batch_scoring.py
```

Data files are gitignored (regenerable via fixed random seed) — clone and run the pipeline above to reproduce everything from scratch.
EOF