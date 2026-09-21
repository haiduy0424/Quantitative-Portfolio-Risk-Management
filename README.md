# Portfolio Risk Analytics Framework

A Python framework for building and risk-managing two USD 50m equity funds benchmarked to the Dow Jones Industrial Average (DJIA).

| Fund | Objective | Approach |
|---|---|---|
| **Active** | Outperform the DJIA on a risk-adjusted basis | Black–Litterman expected returns → constrained max-Sharpe portfolio of 12 stocks |
| **Passive** | Track the DJIA | Full price-weighted replication with index-futures overlays |

**Evaluation period:** 1 Oct 2024 – 20 Nov 2025  **Initial capital:** USD 50m per fund

## 1. Methodology

| Stage | Method |
|---|---|
| **Risk model** | Covariance shrinkage with a cross-validated intensity; compared against sample, constant-correlation and EWMA estimators |
| **Expected returns** | Black–Litterman: market-implied equilibrium prior + absolute views from sell-side target prices |
| **Optimisation** | Maximum Sharpe ratio under a 10% single-stock cap, sector bands, long-only, fully invested |
| **Rebalancing** | Drift-based rule with a ±5% tolerance band around target weights |
| **Factor analysis** | Fama–French 3-factor regression (Mkt-RF, SMB, HML) as a style and risk diagnostic |
| **Market risk** | Monte Carlo simulation (10,000 paths) → 95% / 99% Value-at-Risk |
| **Stress testing** | Portfolio replayed across bullish, bearish and stable historical regimes |
| **Passive fund** | Price-weighted replication including the Nov-2024 index change (INTC, DOW → NVDA, SHW) |
| **Derivatives** | DJIA futures for cash equitization and beta hedging |

## 2. Results

### Active fund vs benchmarks

| Metric | Active fund | DJIA | S&P 500 (SPY) |
|---|---:|---:|---:|
| Total return | **42.12%** | 17.95% | 9.45% |
| Annualised return (CAGR) | **36.45%** | 15.72% | 8.31% |
| Annualised volatility | 20.86% | 18.86% | 16.45% |
| Sharpe ratio | **1.60** | 0.67 | 0.32 |
| Maximum drawdown | -20.59% | -18.76% | -16.37% |

- **Value-at-Risk (1-year, Monte Carlo):**
  - 95%: a USD 40.4m ending value, i.e. a loss of about USD 9.6m (19%)
  - 99%: a USD 38.2m ending value, i.e. a loss of about USD 11.8m (24%)
- **Scenario analysis (2-year total return):** Bullish +41.45%, Bearish -50.61%, Stable +45.58%
- **Factor exposure:** market risk is the dominant driver, size exposure is limited (large-cap universe), and value exposure is modest.

### Passive fund

| Metric | Result |
|---|---:|
| Ending value | USD 55.80m |
| Total return | 11.59% |
| Annualised volatility | 16.45% |
| Sharpe ratio | 0.43 |
| Tracking error vs DJIA | **0.22%** (target < 5%) |
| Maximum drawdown | -15.55% (DJIA: -16.37%) |

### Futures overlays

| Strategy | Effect |
|---|---|
| **Cash equitization** | 6 long DJIA futures deploy the 5% cash buffer, removing cash drag and bringing tracking error close to zero |
| **Beta hedge** | A short futures overlay cuts volatility from 16.21% to 1.47% (return 10.44% → 0.85%), neutralising market exposure |

## 3. Repository structure

```text
├── Code.ipynb                                           # Research notebook: data → construction → risk → results
├── utils.py                                             # Model, backtest, risk and charting functions
├── target_prices_djia.xlsx                              # Analyst 12-month target prices
├── F-F_Research_Data_Factors_daily.xlsx                 # Fama–French daily factors
└── ETF Portfolio Performance_Report_Quant Approach.pdf  # Full written report
```

## 4. Getting started

```bash
git clone <repository-url>
cd Portfolio-Risk-Analytics-Framework
pip install numpy pandas scipy scikit-learn matplotlib openpyxl jinja2 yfinance
jupyter notebook Code.ipynb
```

Run all cells in order. Market data is downloaded from Yahoo Finance.
All parameters (dates, capital, constraints, model settings) live in `ResearchConfig` in `utils.py`.

## 5. Key takeaways

- **Active management** added value within a disciplined framework: a Sharpe ratio of 1.60 vs 0.67 for the DJIA.
- **Black–Litterman** combines market equilibrium with fundamental views in a structured way.
- **Constraints and shrinkage** turn mathematical optimisation into implementable, diversified allocations.
- **Downside risk** varies sharply across market regimes, so a single VaR figure understates tail exposure.
- **Index futures** serve two roles: keeping full benchmark exposure (equitization) or neutralising it (hedging).

## 6. Limitations

- Results depend on the Black–Litterman prior and on analyst views.
- Correlations are unstable across market regimes.
- The model makes simplified distributional assumptions.
- Futures basis risk and roll costs are not fully modelled.

## Disclaimer

Developed for academic purposes. Nothing in this repository is investment advice. Past and simulated performance does not guarantee future results.
