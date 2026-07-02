# Urban Climate AI: Physics-Informed Mitigation Engine

## 📌 Project Overview
This project develops a Geospatial AI/ML pipeline to analyze **Urban Heat Island (UHI)** dynamics. Inspired by remote sensing methodologies, the system diagnoses key thermal drivers (vegetation density, albedo, and morphology) and simulates optimal spatial cooling interventions using an interactive interface.

## 🔬 Core Methodology & Pipeline
1. **Data Integration:** Fusing multi-sensor remote sensing features (LST proxies, Urban Greenness Index) with meteorological constraints.
2. **Predictive Modeling:** Leveraging Gradient Boosting (`XGBoost`) optimized through standardized feature scaling pipelines.
3. **Policy Simulation:** A programmatic feedback loop that alters environmental parameters (e.g., increasing green canopy coverage) to predict localized microclimate temperature drop ($^\circ\text{C}$).

## 🛠️ Tech Stack
* **Language:** Python 3.14+
* **ML Pipeline:** Scikit-Learn, XGBoost, Joblib
* **GUI / Simulation Dashboard:** Streamlit
* **Analytics:** Pandas, NumPy

## 📊 Key Insights for Research
* Quantifies the exact correlation between the *Urban Greenness Ratio* and *Land Surface Temperature*.
* Simulates scenario-based interventions to assist urban planners in strategic tree-canopy allocation.
