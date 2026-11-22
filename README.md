# AppGuard AI: Insurance Application Fraud Detector

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)]()

**AppGuard AI** is an unsupervised anomaly detection research prototype and specialized insurance operations review workbench. It learns the multidimensional manifold of typical, nominal policy applications using a PyTorch Variational Autoencoder (VAE) and computes calibrated reconstruction divergence scores to triage atypical applications for operational human underwriting review.

---

## 🏛️ Purpose and Boundaries

> [!IMPORTANT]
> **Advisory Review Signal Only:** Anomaly scores quantify statistical divergence from nominal application profiles. An elevated anomaly score does **not** prove fraud, misrepresentation, or deceptive intent.
>
> **Adverse Action Boundaries:** Scores are advisory triage signals only. Automated approval, rejection, denial of coverage, adverse premium surcharges, or cancellation based solely on model output must not occur. Applicable legal and regulatory requirements regarding automated decisioning vary by jurisdiction and intended use; real deployments require qualified legal and compliance review. Qualified human underwriters must perform all final adjudications.
>
> **Privacy-First Design:** Strict exclusion of names, government IDs, full birth dates, street addresses, ZIP code proxies, and protected demographic traits (race, religion, sex, marital status, disability). Zero external telemetry.

---

## 📚 Educational Project Deliverables

This repository contains four comprehensive project deliverables:

1. **[Deliverable 1: Insurance Application Dataset Preparation Report](deliverables/01_insurance_application_dataset_preparation_report.md)**  
   *Schema specification, privacy boundaries, multivariate synthetic data generation, preprocessing, handling missing values, and dataset limitations.*
2. **[Deliverable 2: Variational Autoencoder Anomaly Detection Model](deliverables/02_variational_autoencoder_anomaly_detection_model.md)**  
   *PyTorch VAE mathematical formulation, neural architecture, training regimen, reconstruction error scoring, empirical 95th percentile calibration, and artifact bundling.*
3. **[Deliverable 3: AppGuard AI Fraud Detection API Reference](deliverables/03_fastapi_scoring_api.md)**  
   *FastAPI service architecture, REST endpoint contracts, Pydantic schemas, safe per-record batch error isolation, structured logging without PII, and run commands.*
4. **[Deliverable 4: Final AppGuard AI Project Report](deliverables/04_final_appguard_ai_project_report.md)**  
   *End-to-end methodology synthesis, quantitative benchmark evaluation on 1,000 synthetic applications, synthetic demo disclaimers, ethical constraints, and future technical roadmap.*

---

## 🔬 Synthetic Benchmark Evaluation

> [!IMPORTANT]
> These figures measure the VAE's ability to separate two **synthetic distributions defined by this project's own data generator** — not performance on real insurance applications or real fraud. The "anomaly" labels are assigned programmatically, not by fraud investigators. These numbers must not be cited as evidence of real-world fraud-detection capability.

| Metric | Synthetic Result | Synthetic Target | Scope |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9904** | $\ge 0.9000$ | Synthetic distributions only |
| **Recall (Sensitivity)** | **97.00%** (194 / 200) | $\ge 90.00$ | Against generator-assigned labels |
| **Precision** | **84.72%** (194 / 229) | $\ge 80.00$ | Against generator-assigned labels |
| **F1-Score** | **0.9044** | $\ge 0.8500$ | Synthetic evaluation only |
| **False Positive Rate on Nominal** | **4.37%** (35 / 800) | $\le 5.00$ | Calibration consistency check |

---

## ⚡ Quickstart & Installation

### 1. Local Environment Setup
```bash
# Clone or navigate to the workspace
cd appguard-ai

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest -v
# 17 automated tests covering preprocessing, VAE architecture, calibration,
# API validation, safe batch error isolation, and demo flow.
```

### 3. Generate Datasets & Train Model (Optional — Pre-bundled)
```bash
# Generate synthetic dataset splits in data/
python -m appguard.data_generator

# Train PyTorch VAE and package versioned artifact bundle
python -m appguard.calibration
```

### 4. Launch Service & Workbench
```bash
# Launch unified API & Operations Workbench on port 8000
uvicorn appguard.api:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to interact with the Insurance Operations Workbench.

---

## 🐳 Docker Deployment

The application includes a non-root production Docker configuration:

```bash
# Build and run containerized service
docker compose up --build -d

# Verify container health check
docker compose ps
curl http://127.0.0.1:8000/api/v1/health
```

---

## 🌐 API Reference & Verification Examples

### Health Check
```bash
curl -s http://127.0.0.1:8000/api/v1/health | jq .
```

### Governance Metadata
```bash
curl -s http://127.0.0.1:8000/api/v1/model-info | jq .
```

### Score Individual Application
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "DEMO-EXP-01",
    "policy_type": "term_life",
    "coverage_amount": 2500000,
    "deductible_amount": 0,
    "premium_frequency": "monthly",
    "application_channel": "online_direct",
    "applicant_age": 22,
    "annual_income_estimate": 19000,
    "prior_application_count": 0,
    "recent_address_changes_count": 1,
    "property_or_vehicle_age": 0,
    "beneficiary_relation": "other",
    "declared_credit_tier": "limited_history"
  }' | jq .
```

---

## 🖥️ Insurance Operations Workbench UI

The web interface is designed as a quiet, editorial underwriting workbench:
- **Warm off-white palette:** `#fbfaf7` page, `#ffffff` surface, fine `#d4cdc0` borders.
- **Editorial typography:** Newsreader serif headings, clean humanist body typography, tabular monospace numerals.
- **Restrained status badges:** Muted forest green for standard routing; restrained amber for review recommendations.
- **Queue Table & Detail Panel:** Two-pane inspection workflow displaying submitted parameters alongside decoded feature-level statistical deviations.
- **Interactive Underwriter Adjudication:** Operational action buttons (*Clear for Standard Underwriting*, *Request Verification Documents*, *Refer to SIU*) with audit rationale logging.
- **Synthetic Demo Mode:** One-click simulation to generate fresh batches of applications or evaluate curated test scenarios.

---

## 🛡️ License & Ethics

Educational and research prototype. Scores must not be used for automated adverse decisions; applicable legal and regulatory requirements vary by jurisdiction and intended use. No legal or regulatory advice provided. Built with zero external telemetry and 100% synthetic data.
