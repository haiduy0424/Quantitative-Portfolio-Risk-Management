# Active and Passive Equity Management on the DJIA: A Systematic Quantitative Approach

This project represents an end-to-end portfolio research pipeline in Python that builds 2 long-only funds benchmarked against the Dow Jones Industrial Average (DJIA):

- **Active fund:** a quantitative stock-selection and allocation strategy built on a Bayesian return model and constrained optimization.
- **Passive fund:** a full index-replication fund that uses index futures for cash management and hedging.

All models are estimated on an in-sample window and evaluated on a separate, unseen out-of-sample period to avoid look-ahead bias.

## 1. Scope

| Area | What the project covers |
|---|---|
| **Research design** | In-sample / out-of-sample split, benchmark selection (DJIA, SPY), point-in-time universe with index reconstitution |
| **Risk model** | Ledoit–Wolf shrinkage covariance, with the shrinkage intensity chosen by cross-validated out-of-sample log-likelihood |
| **Return forecasting** | Black–Litterman model: market-implied equilibrium prior, analyst target-price views, Idzorek confidence-based view uncertainty |
| **Portfolio construction** | Max-Sharpe mean–variance optimization under long-only, sector and single-stock constraints, with a concentrated high-conviction portfolio |
| **Backtesting** | Drift-band rebalancing backtest with transaction costs and turnover tracking |
| **Risk analysis** | Monte Carlo VaR, historical and parametric VaR/CVaR, drawdown analysis, and scenario tests across bull, bear, stable and crisis regimes |
| **Performance attribution** | Fama–French three-factor regression to separate market, size and value exposures from stock-specific return |
| **Index replication** | Price-weighted replication of the DJIA, reconstitution handling, tracking-error measurement |
| **Derivatives overlay** | Futures-based cash equitization to remove cash drag, and a short-futures beta hedge to control market exposure |

## 2. Technical Skills Demonstrated

- **Statistical estimation:** covariance shrinkage, cross-validation, Bayesian updating of expected returns.
- **Optimization:** constrained mean–variance and Sharpe-ratio optimization.
- **Risk modeling:** simulation-based VaR, tail-risk measures, regime-based stress testing.
- **Factor modeling:** OLS factor regressions for return and risk attribution.
- **Backtesting:** out-of-sample evaluation, rebalancing rules, transaction-cost modeling, risk-adjusted performance metrics.
- **Derivatives:** index futures for equitization and hedging.
- **Engineering:** modular, reusable Python backend (data loading, estimators, optimizers, backtester, analytics) driven from a single research notebook.

## 3. Tech Stack

`Python` · `NumPy` · `pandas` · `SciPy` · `scikit-learn` · `statsmodels` · `PyPortfolioOpt` · `Matplotlib` · `Seaborn` · `Plotly` · `yfinance`

## 4. My Role

- Co-first author in a five-member team, contributing equally to the research and report.
- Led the quantitative modeling and built the end-to-end Python pipeline, from risk modeling and optimization to backtesting.

## 5. Repository Structure

```
├── Main.ipynb                           # Research notebook: full pipeline from data to evaluation
├── utils.py                             # Backend: data fetching, covariance, Black–Litterman, backtesting, risk, passive fund
├── target_prices_djia.xlsx              # Analyst target prices used as Black–Litterman views
├── F-F_Research_Data_Factors_daily.xlsx # Fama–French daily factors
└── Report.pdf                           # Full written report with results and discussion
```

## Disclaimer

This project is intended for educational purposes only and does not constitute investment advice.
