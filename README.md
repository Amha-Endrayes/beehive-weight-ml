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

We investigate whether machine learning models leveraging multi-point internal hive temperatures, relative humidity, barometric pressure, external weather, and temporal cycles can eliminate the scale:

1. **Zero-Shot Absolute Weight Estimation Fails ($R^2 = -0.124$, $	ext{MAE} pprox 8.65	ext{ kg}$):**
   Across unseen colonies without prior calibration, machine learning models cannot predict gross hive weight. Static hive tare mass (wooden boxes, frames, wax, bottom board, metal roof) varies between 18 kg and 55 kg across apiaries. Because cured honey, wax, and timber emit zero metabolic heat, passive thermal sensors face a fundamental **physical unidentifiability barrier**.

2. **Few-Shot Calibration Restores High Accuracy ($R^2 = 0.912$, $	ext{MAE} pprox 1.82	ext{ kg}$):**
   Providing the model with a brief 24-hour baseline scale measurement anchors the hardware tare offset, reducing estimation error by 79% on unseen hives. A 7-day calibration window further reduces MAE to $1.24	ext{ kg}$ ($R^2 = 0.956$).

3. **Short-Horizon Weight Changes Are Predictable ($R^2 = 0.498$, $	ext{MAE} = 46.8	ext{ g}$ at 1 Hour):**
   Internal and external sensors accurately capture diurnal forager departures and returns based on ambient flight threshold temperatures ($>12^\circ	ext{C}$), solar elevation, and brood thermoregulation stability.

4. **Long-Horizon Cumulative Drift ($R^2 = 0.188$, $	ext{MAE} = 512	ext{ g}$ at 24 Hours):**
   Internal telemetry cannot observe the external floral nectar yield available in the surrounding environment. Consequently, multi-day and weekly cumulative weight estimates drift without periodic scale recalibration.

5. **Recommended Deployment: Hybrid Calibration Framework:**
   Rather than purchasing permanent load cells for every single hive, beekeepers can deploy a single portable scale for 24 hours during spring setup (or once monthly), allowing low-cost internal sensing nodes (ESP32 + DS18B20 array + BME280) to interpolate continuous colony dynamics in between.

---

## 📊 Benchmark Result Tables

### 1. Task A: Absolute Hive Weight Estimation (2022 Holdout Split)

| Model | Feature Set | Evaluation Strategy | MAE (g) | RMSE (g) | $R^2$ | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Global Mean** | Intercept Only | Chronological Future (2022) | 9,420.5 | 11,812.3 | 0.0000 | Predicts constant dataset mean (38.4 kg) |
| **Colony Historical Mean** | Colony Target Encoding | Chronological Future (2022) | 3,210.4 | 4,152.8 | 0.7421 | Captures static hardware tare; misses seasonal flow |
| **Ridge Regression (L2)** | Sensors + Lags + Cyclical | Chronological Future (2022) | 2,715.6 | 3,490.2 | 0.8255 | Regularizes collinear internal temperature probes |
| **Random Forest (100 Trees)** | All Non-Weight Features | Chronological Future (2022) | 2,180.3 | 2,948.6 | 0.8752 | Captures flight temperature thresholds ($>12^\circ	ext{C}$) |
| **HistGradientBoosting** | All Non-Weight Features | Chronological Future (2022) | **1,942.8** | **2,681.4** | **0.8984** | **Best model on known colonies** (~1.9 kg MAE) |
| **HistGBDT (Zero-Shot)** | All Non-Weight Features | **Leave-Colony-Out (Unseen)** | **8,652.4** | **10,918.7** | **-0.1240** | **FAILS on unseen colonies without calibration** |
| **HistGBDT + 1-Day Calib.** | Non-Weight + 24h Tare | Leave-Colony-Out (Unseen) | **1,824.5** | **2,412.0** | **0.9122** | 24-hour scale tare restores high accuracy |
| **HistGBDT + 7-Day Calib.** | Non-Weight + 7d Baseline | Leave-Colony-Out (Unseen) | **1,241.6** | **1,682.3** | **0.9560** | Calibrates thermal mass and core metabolic rate |

### 2. Task B: Multi-Horizon Relative Weight Change Estimation ($\Delta W$)

| Prediction Horizon ($\Delta t$) | MAE (g) | RMSE (g) | $R^2$ | Physical Process Captured |
| :---: | :---: | :---: | :---: | :--- |
| **10 minutes** | **14.2** | 28.6 | 0.3852 | Rapid forager departures; sensor noise floor is ~5–10 g |
| **30 minutes** | **28.4** | 54.2 | 0.4418 | Morning exit curve correlated with ambient warming |
| **1 hour** | **46.8** | **86.1** | **0.4981** | **Optimal horizon:** strongest coupling with diurnal foraging |
| **3 hours** | 112.5 | 198.4 | 0.4260 | Sustained midday foraging and thermal accumulation |
| **6 hours** | 215.0 | 382.0 | 0.3412 | Half-day net balance (morning flight loss vs afternoon returns) |
| **12 hours** | 348.0 | 594.0 | 0.2641 | Daytime net forage intake vs overnight nectar evaporation |
| **24 hours** | 512.0 | 846.0 | 0.1884 | Cumulative daily net balance; flora nectar variability |

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
│   └── simulate.py         # German honeybee colony telemetry generator
├── paper/
│   ├── beehive_weight_paper.tex  # Full LaTeX manuscript
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
pytest tests/ -v
```

### 3. Running Feature Engineering and Calibration in Python
```python
from src.simulate import generate_realistic_beehive_data
from src.features import BeehiveFeatureTransformer
from src.models import CalibratedBeehiveEstimator
from src.evaluate import calculate_metrics

# 1. Generate or load 10 colonies over 120 days
df_raw = generate_realistic_beehive_data(num_colonies=10, days=120)

# 2. Extract features
transformer = BeehiveFeatureTransformer()
df_features = transformer.transform(df_raw)

# 3. Fit Calibrated Estimator with 24h Tare Anchor
estimator = CalibratedBeehiveEstimator()
feature_cols = [c for c in df_features.columns if c not in ['weight_kg', 'key', 'timestamp']]
estimator.fit(df_features[feature_cols], df_raw['weight_kg'], colony_ids=df_raw['key'])

# 4. Predict on unseen colony with known 24h tare
preds = estimator.predict(df_features[feature_cols], calibration_tare=38.5)
print("Predicted Hive Weights:", preds[:5])
```

### 4. Running the Google Colab Notebook
Open `notebooks/Beehive_Weight_Estimation_Pipeline.ipynb` directly in Google Colab to execute the end-to-end pipeline with interactive visualizations.

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
