# Active and Passive Equity Management on the DJIA: A Systematic Quantitative Approach

This project builds and evaluates two DJIA-benchmarked long-only equity funds (an active fund using Bayesian return forecasting and constrained optimization, and a passive index-replication fund with a derivatives overlay) assessed across in-sample and out-of-sample periods using rigorous risk and performance metrics.

## 1. Supervisor

Nguyen Thi Hoang Anh, PhD — Foreign Trade University, HCMC Campus

## 2. Timeline

Project start to end: December 2025

## 3. Scope of Work

* Construct an **Active Fund** (12 high-conviction DJIA stocks) using Black–Litterman return forecasting and max-Sharpe optimization under sector and position-size constraints.
* Construct a **Passive Fund** replicating the DJIA via price-weighted holdings with event-triggered reconstitution and a futures overlay for cash equitization and beta management.

Perform in-sample (IS) and out-of-sample (OS) analyses:

* IS: 2019-01-01 to 2022-01-01
* OS: 2022-01-01 to 2025-11-20

Predict and evaluate portfolio risk and performance using statistical and econometric models.

## 4. Methodology

### 4.1. Data Collection

* Daily DJIA constituent prices retrieved via Yahoo Finance or equivalent APIs.
* Fama–French daily factors sourced from WRDS.
* Analyst target prices collected from JPMorgan, Barclays, and Wells Fargo research.

### 4.2. Portfolio Construction

* Active Fund: Black–Litterman Absolute Views blending market-implied equilibrium returns with analyst views; max-Sharpe optimization under long-only, sector-budget, and single-stock ≤ 10% constraints.
* Passive Fund: price-weighted replication of the DJIA; event-triggered rebalancing on index reconstitution; DJIA E-mini futures for cash equitization and beta overlay.
* Returns, standard deviation, Sharpe Ratio, and correlation matrix calculated for both funds.

### 4.3. Statistical Analysis

* Asset-wise return distributions analyzed across IS and OS periods.
* Portfolio return and volatility calculated using time-series data.
* Scenario analysis conducted across bullish, bearish, and stable historical regimes.

### 4.4. Risk Modeling

* Risk estimated and predicted using:
  * **Ledoit–Wolf shrinkage covariance** — reduces estimation error in the sample covariance matrix.
  * **Black–Litterman model** — Bayesian framework producing stable forward-looking return expectations.
  * **Monte Carlo simulation** — generates full distribution of terminal portfolio values and VaR/CVaR.
  * **Fama–French three-factor regression** — decomposes returns into market, size, and value exposures.
* Models compared using performance on out-of-sample data.

## 5. Key Results & Insights

### 5.1. Active Fund Performance

* The Active Fund significantly outperformed both the DJIA benchmark and S&P 500 in total and risk-adjusted returns over the evaluation period.
* Zero rebalancing events were triggered — initial weights remained within the ±5% drift band throughout, reflecting a stable and well-diversified initial allocation.
* Fama–French regression confirms a quality large-cap tilt with dominant market-risk loading and limited size/value exposure.

### 5.2. Scenario & Downside Analysis

* Under a bullish regime, the Active Fund delivers strong gains, broadly in line with the passive strategy.
* Under a bearish regime, the concentrated active tilts amplify drawdowns significantly, highlighting the asymmetric downside risk of active management.
* These results underscore the importance of regime awareness in risk budgeting and position sizing.

### 5.3. Passive Fund & Derivatives Overlay

* The Passive Fund closely tracks the DJIA with a tracking error well within the 5% target.
* Cash equitization via DJIA E-mini futures contracts effectively eliminates cash drag, reducing tracking error to near zero.
* The short-futures beta overlay compresses volatility but introduces benchmark divergence — illustrating the trade-off between capital preservation and index replication.

## 6. Disclaimer

This project is intended for educational purposes only and does not constitute investment advice.
