# House Price Prediction — Intelligent ML Valuation Engine

A high-performance machine learning regression project that accurately predicts residential real-estate prices across premier Indian metropolitan markets. Features a standalone modern glassmorphic web dashboard with real-time valuation, archetype presets, and interactive Chart.js analytics.

---

## 🌟 Key Features

- **End-to-End ML Pipeline:** Scikit-Learn pipeline incorporating median imputation, `StandardScaler`, and robust `OneHotEncoder` handling unseen categories.
- **Multiple Regression Architectures:** Benchmarks Linear Regression, Random Forest, and Gradient Boosting Regressors.
- **High Precision:** Gradient Boosting achieves $R^2 \approx 0.9884$ (98.8% variance explained) with MAE $\approx 10.99$ Lakhs.
- **Standalone Web Application:** Built with semantic HTML5, Vanilla CSS3 (dark glassmorphism, Google Fonts), and JavaScript (no framework overhead).
- **1-Click Archetype Presets:** Instant configuration for *Compact Studio*, *Family Residence*, *Luxury Penthouse*, and *Suburban Villa*.
- **Interactive Market Intelligence:** Responsive Chart.js charts comparing city average prices and area vs. price distributions.
- **Automated Test Suite:** Comprehensive test coverage via `pytest`.

---

## 🏗️ Project Architecture

```text
House-Price-Prediction-ML/
├── app.py                     # Flask web server & REST API (/api/predict, /api/market-data)
├── train_model.py             # Model training, comparison & persistence
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes venv, cache & temp files
├── dataset/
│   └── house_prices.csv       # Training and evaluation dataset
├── models/
│   ├── house_price_model.pkl  # Serialized best model pipeline
│   └── model_results.json     # Model comparison leaderboard metrics
├── templates/
│   └── index.html             # Semantic HTML5 frontend dashboard
├── static/
│   ├── css/
│   │   └── style.css          # Vanilla CSS luxury glassmorphism design system
│   └── js/
│       └── app.js             # Client-side reactivity, presets & Chart.js integration
├── notebooks/
│   └── house_price_analysis.ipynb
├── docs/                      # Internship & project documentation
└── tests/
    └── test_project.py        # Automated validation test cases
```

---

## ⚡ Quick Start

### 1. Set Up Environment
```bash
python -m venv venv
```

**Activate Environment:**
- Windows (PowerShell):
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows (CMD):
  ```cmd
  venv\Scripts\activate.bat
  ```
- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train Models
```bash
python train_model.py
```

### 4. Run the Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🧪 Testing

Execute automated unit tests:
```bash
pytest -v
```

---

## 📊 Model Leaderboard

| Model | $R^2$ Score | MAE (Lakh ₹) | RMSE (Lakh ₹) |
|---|---|---|---|
| **Gradient Boosting (Production)** | **0.9884** | **10.99** | **14.08** |
| Linear Regression | 0.9830 | 12.91 | 17.06 |
| Random Forest | 0.9639 | 18.62 | 24.87 |

---

## 📜 License
Educational & research project for machine learning workflows and full-stack integration.
