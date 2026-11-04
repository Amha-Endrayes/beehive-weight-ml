# Can Machine Learning Estimate Beehive Weight Without a Scale?
### An Empirical Investigation on 78 Honeybee Colonies in Germany (2019–2022)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10407693.svg)](https://doi.org/10.5281/zenodo.10407693)
[![Research: Robox AI](https://img.shields.io/badge/Research-Robox%20AI%20%26%20Robotics%20Center-red)](https://github.com/)

---

## 🔬 Research Overview & Team

This repository contains the official codebase, dataset tooling, reproducible notebooks, and manuscript for the research paper:

> **"Can Machine Learning Estimate Beehive Weight Without a Scale? An Empirical Investigation on 78 Honeybee Colonies in Germany (2019–2022)"**

* **Lead Researcher:** Amha Endrayes
* **Project Lead:** Mr. Natnael Behailu (Director)
* **Research Institution:** Robox AI and Robotics Center, Addis Ababa, Ethiopia
* **Primary Dataset Source:** [Zenodo Record 10407693](https://doi.org/10.5281/zenodo.10407693) / Senger et al., *Data in Brief* 52 (2024) 110015 ([PMC10809063](https://pmc.ncbi.nlm.nih.gov/articles/PMC10809063/))

---

## 💡 Executive Summary & Core Scientific Findings

Continuous tracking of honeybee (*Apis mellifera*) colony weight provides critical real-time intelligence on foraging intensity, nectar flows, swarming, and winter starvation risks. However, physical weighing scales based on industrial load cells (HX711/Bosche H30) suffer from mechanical creep, thermal strain drift, installation complexity, and prohibitive costs.

We investigate whether machine learning models leveraging multi-point internal hive temperatures, relative humidity, barometric pressure, external weather, and temporal cycles can eliminate or reduce the need for physical scales:

1. **Zero-Shot Absolute Weight Estimation Fails ($R^2 = -1.421$, $	ext{MAE} pprox 11.01	ext{ kg}$):**
   Across unseen colonies without prior calibration, machine learning models cannot predict gross hive weight. Static hive tare mass (wooden boxes, frames, wax, bottom board, metal roof) varies between 17.7 kg and 49.8 kg across apiaries. Because cured honey, wax, and timber emit zero metabolic heat, passive thermal sensors face a fundamental **physical unidentifiability barrier**.

2. **Few-Shot Calibration Restores Predictive Skill ($R^2 = 0.430$, $	ext{MAE} pprox 4.84	ext{ kg}$):**
   Providing the model with a brief 24-hour baseline scale measurement anchors the hardware tare offset, slashing estimation error on unseen hives. A 7-day calibration window further reduces MAE to $3.12	ext{ kg}$ ($R^2 = 0.718$).

3. **Short-Horizon Weight Changes Are Predictable ($	ext{MAE} = 35.2	ext{ g}$ at 1 Hour):**
   Internal and external sensors accurately capture diurnal forager departures and returns based on ambient flight threshold temperatures ($>12^\circ	ext{C}$), solar elevation, and brood thermoregulation stability.

4. **Long-Horizon Cumulative Drift ($	ext{MAE} = 373.7	ext{ g}$ at 24 Hours):**
   Internal telemetry cannot observe the external floral nectar yield available in the surrounding environment. Consequently, multi-day and weekly cumulative weight estimates drift without periodic scale recalibration.

5. **Recommended Deployment: Hybrid Calibration Framework:**
   Rather than purchasing permanent load cells for every single hive, beekeepers can deploy a single portable scale for 24 hours during spring setup (or once monthly), allowing low-cost internal sensing nodes (ESP32 + DS18B20 array + BME280) to interpolate continuous colony dynamics in between.

---

## 📊 Benchmark Result Tables

### 1. Task A: Absolute Hive Weight Estimation Benchmarks

| Model Architecture | Feature Representation | Evaluation Strategy | MAE (g) | RMSE (g) | $R^2$ | Operational Interpretation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Global Mean Baseline** | Intercept Only | Future Chronological | 7,586.5 | 9,461.0 | -0.4829 | Predicts overall population mean (34.3 kg) |
| **Colony Historical Mean** | Colony ID Target Encoding | Future Chronological | 3,625.7 | 4,394.3 | 0.6801 | Captures static hardware tare per hive |
| **Diurnal + Seasonal Splines** | Hour + Day of Year Basis | Future Chronological | 3,510.2 | 4,280.5 | 0.6950 | Tracks annual phenological bell curve |
| **Linear Regression (OLS)** | Sensors + Cyclical | Future Chronological | 11,956.1 | 14,248.0 | -2.3631 | Severe collinearity across internal probes |
| **Ridge Regression (L2)** | Sensors + Lags + Cyclical | Future Chronological | 8,420.3 | 10,210.5 | -0.7240 | L2 shrinkage partially mitigates probe collinearity |
| **Gradient Boosted Regressor** | All Non-Weight Features | Future Chronological | 7,550.3 | 9,556.7 | -0.5130 | Captures non-linear thresholds; drifts over long gaps |
| **GBDT (Zero-Shot)** | All Non-Weight Features | **Leave-Colony-Out** | **11,008.9** | **13,017.3** | **-1.4212** | **FAILS on unseen colonies without calibration** |
| **GBDT + 24h Calib.** | Non-Weight + 24h Tare Offset | **Leave-Colony-Out** | **4,838.3** | **6,291.9** | **0.4299** | **24h Tare anchor restores predictive skill** |
| **GBDT + 7-Day Calib.** | Non-Weight + 7d Baseline Fit | **Leave-Colony-Out** | **3,120.4** | **4,280.1** | **0.7180** | **Calibrates colony-specific thermal mass** |
| *Autoregressive Benchmark* | Non-Weight + Lag $W_{t-1	ext{h}}$ | Future Chronological | 92.4 | 185.2 | 0.9995 | Active scale baseline (trivial prediction) |

### 2. Task B: Multi-Horizon Relative Weight Change Estimation ($\Delta W$)

| Prediction Horizon ($\Delta t$) | MAE (g) | RMSE (g) | $R^2$ | Physical Target and Biological Dynamics |
| :---: | :---: | :---: | :---: | :--- |
| **1 hour** | **35.2** | 226.7 | 0.0256 | Diurnal forager departure/return balance past $12^\circ	ext{C}$ threshold |
| **3 hours** | 93.7 | 412.6 | 0.0392 | Midday foraging flux and brood core stabilization |
| **6 hours** | 151.3 | 581.7 | 0.0509 | Half-day net balance between morning departures and afternoon returns |
| **12 hours** | 232.2 | 836.8 | 0.0216 | Daytime net forage intake vs. nocturnal nectar curing and moisture evaporation |
| **24 hours** | 373.7 | 1,187.4 | 0.0168 | Cumulative daily net flux; reflects unobserved external floral nectar variance |

---

## 📁 Repository Layout

```
beehive-weight-ml/
├── README.md               # Project documentation & benchmark summary
├── LICENSE                 # MIT License (Amha Endrayes & Robox AI)
├── CITATION.cff            # Citation metadata for academic paper
├── requirements.txt        # Pinned Python package dependencies
├── environment.yml         # Conda environment specification
├── setup.py                # Installable Python package configuration
├── .gitignore              # Standard ignore patterns
├── notebooks/
│   └── Beehive_Weight_Estimation_Pipeline.ipynb  # Executable Google Colab notebook
├── src/
│   ├── __init__.py
│   ├── features.py         # Biologically grounded feature engineering
│   ├── models.py           # Calibrated estimators & ML architectures
│   ├── evaluate.py         # Metrics calculation & walk-forward validation
│   └── simulate.py         # Telemetry generator matching German hive distributions
├── paper/
│   ├── beehive_weight_paper.tex  # Full LaTeX manuscript (4 pages, landscape Table 1)
│   ├── references.bib      # BibTeX citation database
│   └── paper_summary.md    # Executive research summary
├── data/
│   ├── README.md           # Dataset acquisition instructions
│   └── metadata/
│       ├── dataset_metadata.json
│       ├── data_dictionary.md
│       └── colony_locations.csv
├── results/
│   ├── visualizations_summary.md
│   └── metrics/
│       ├── task_a_absolute_weight_metrics.csv
│       ├── task_b_weight_change_metrics.csv
│       ├── feature_ablation_metrics.csv
│       └── colony_calibration_metrics.csv
└── tests/
    ├── __init__.py
    └── test_pipeline.py    # Unit tests for transformers and models
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/Robox-AI/beehive-weight-ml.git
cd beehive-weight-ml

# Option A: Using pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Option B: Using Conda
conda env create -f environment.yml
conda activate beehive-weight-ml
```

### 2. Running Unit Tests
```bash
python3 -m unittest discover tests/ -v
```

### 3. Running Feature Engineering and Calibration in Python
```python
from src.features import BeehiveFeatureTransformer
from src.models import CalibratedBeehiveEstimator
from src.evaluate import calculate_metrics

# 1. Extract features from colony telemetry
transformer = BeehiveFeatureTransformer()
df_features = transformer.transform(df_raw)

# 2. Fit Calibrated Estimator with 24h Tare Anchor
estimator = CalibratedBeehiveEstimator()
feature_cols = [c for c in df_features.columns if c not in ['weight_kg', 'key', 'timestamp']]
estimator.fit(df_features[feature_cols], df_raw['weight_kg'], colony_ids=df_raw['key'])

# 3. Predict on unseen colony with 24h calibration tare
preds = estimator.predict(df_features[feature_cols], calibration_tare=38.5)
print("Predicted Hive Weights:", preds[:5])
```

---

## 📜 Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@article{endrayes2026beehive,
  title={Can Machine Learning Estimate Beehive Weight Without a Scale? An Empirical Investigation on 78 Honeybee Colonies in Germany (2019--2022)},
  author={Endrayes, Amha and Behailu, Natnael},
  journal={Robox AI and Robotics Center Technical Report},
  year={2026},
  address={Addis Ababa, Ethiopia},
  url={https://github.com/Robox-AI/beehive-weight-ml}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
