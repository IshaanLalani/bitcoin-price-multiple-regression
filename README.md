# Bitcoin Price Analysis (Multiple Linear Regression)

An empirical data science project investigating the dynamic structural relationship between Bitcoin's spot price and multi-dimensional blockchain ledger and market indicators, utilizing a Multiple Linear Regression (MLR) framework.

* **Authors:** Obaidullah, Ishaan Lalani, Shayan Bin Ali, Yassal  
* **Institution:** DHA Suffa University (Course Instructor: Sir Faisal)  

---

## 📌 Implementation Note: SPSS & Python
While the foundational academic research, ANOVA testing, and diagnostic checks for this university project were performed using **IBM SPSS Statistics**, this repository provides the equivalent **Python implementation** (`statsmodels` & `pandas`). This bridges enterprise statistical workflows with clean, reproducible code suitable for modern developer portfolios.

---

## 📂 Research Overview & Variables

The study analyzes a sequential blockchain dataset consisting of 30 daily observations ($N = 30$) across distinct market regimes.

* **Dependent Variable (Y):** Bitcoin Closing Price ($)[cite: 3]
* **Independent Variables ($X$):**
  1. **Trading Volume ($B):** Indicator of market liquidity and investor participation[cite: 3].
  2. **Market Capitalization ($B):** Aggregated monetary valuation of the asset[cite: 3].
  3. **Hash Rate (EH/s):** Computational power securing the Proof-of-Work network[cite: 3].
  4. **Circulating Supply (M):** Total volume of minted tokens reflecting asset scarcity[cite: 3].

---

## ⚙️ Methodology & Statistical Framework

1. **Preprocessing:** Applied Min-Max Normalization to scale all independent predictor variables between $0$ and $1$[cite: 3].
2. **Model Estimation:** Constructed an Ordinary Least Squares (OLS) Multiple Linear Regression model[cite: 3]:
   $$\text{Bitcoin Price} = \beta_0 + \beta_1(\text{Trading Volume}) + \beta_2(\text{Market Cap}) + \beta_3(\text{Circulating Supply}) + \beta_4(\text{Hash Rate})$$
3. **Diagnostics & Testing:** Evaluated model significance via ANOVA ($F$-statistic, $p < 0.001$), checked residual normality, and tested for multicollinearity using Variance Inflation Factor (VIF) diagnostics[cite: 3].

---

## 💻 Code Overview & Implementation

The accompanying Python script (`bitcoin_regression.py`) replicates the empirical analysis programmatically:
* **Data Processing:** Builds the time-series data frame utilizing historical ledger metrics.
* **OLS Modeling:** Utilizes `statsmodels.api` to fit the regression equation, generate standard error statistics, and output coefficient significance tables.
* **Multicollinearity Testing:** Computes VIF values across predictor metrics to identify collinearity boundaries.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data Input
        V[Trading Volume] --> MLR
        MC[Market Capitalization] --> MLR
        HR[Hash Rate] --> MLR
        CS[Circulating Supply] --> MLR[Multiple Linear Regression Model OLS]
    end

    MLR --> Stat[Statistical Diagnostics SPSS & Python]
    Stat --> ANOVA[ANOVA & Significance Testing]
    Stat --> VIF[Multicollinearity VIF Analysis]
    ANOVA --> Res[Bitcoin Price Prediction & Insights]
    VIF --> Res
