#  Telecom Customer Churn Analysis & Prediction

A machine learning project that analyzes customer churn patterns in a telecommunications dataset and builds a predictive model to identify customers at risk of leaving.

---

##  Objectives

1. **Exploratory Analysis** — Identify patterns and key differences between customers who churn and those who stay, using the IBM Telco Customer Churn dataset.
2. **Predictive Modeling** — Train a classification model that predicts whether a customer will churn based on their service interactions and demographics.

---

##  Project Structure

```
customer-churn/
│
├── notebooks/
│   └── Customer-Churn.ipynb     # Main analysis & modelling notebook
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (add manually)
│
├── models/                      # Saved model files (generated after training)
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

##  Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains information about:
- **Demographics** — gender, senior citizen status, partner, dependents
- **Services** — phone, internet, online security, backup, device protection, tech support, streaming TV & movies
- **Account info** — contract type, payment method, paperless billing, monthly & total charges, tenure
- **Target** — `Churn` (Yes / No)

> **Note:** Download the dataset from the link above and place the CSV file inside the `data/` folder before running the notebook.

---

##  Key Findings

| Factor | Churn Insight |
|---|---|
| Contract Type | Month-to-month customers churn at the highest rate |
| Internet Service | Fiber optic users churn at ~42% vs DSL at ~19% |
| Tech Support | Customers without tech support churn at ~42% |
| Senior Citizens | 42% churn rate vs 24% for non-seniors |
| Partners | Single customers churn more than those with partners |
| Payment Method | Electronic check users have the highest churn rate |

---

##  Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Primary classifier (with L2 regularisation) |
| Random Forest | Feature importance analysis |

**Feature Engineering:**
- `tenure_year` — tenure converted from months to years
- One-hot encoding for all categorical columns
- `StandardScaler` applied to numerical features

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn.git
cd customer-churn
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the file at:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Run the notebook
```bash
jupyter notebook notebooks/Customer-Churn.ipynb
```

---

##  Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computing |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisations |
| scikit-learn | ≥ 1.3 | ML models & preprocessing |
| jupyter | ≥ 1.0 | Notebook environment |

Install all with:
```bash
pip install -r requirements.txt
```

---

## Recommendations

Based on the analysis, the following strategies are suggested for reducing churn:

- **Promote longer contracts** — Customers on month-to-month plans churn far more; offer incentives for annual or 2-year contracts.
- **Improve fiber optic experience** — The 42% churn rate among fiber optic users suggests service quality or pricing issues.
- **Upsell tech support** — Customers without tech support churn at nearly 3x the rate of those with it.
- **Target senior citizens** — Tailored plans or onboarding support could improve retention in this segment.
- **Engage single customers** — Female customers without partners show a 34.4% churn rate; targeted retention offers may help.

---

##  Author

**Obi Osita**
- GitHub: [github.com/obioraosita18-afk

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
