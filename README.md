# Active & Passive DJIA Fund Management: Black–Litterman Optimisation, Risk Modelling and Futures Overlays

A Python framework for building and risk-managing two USD 50m equity funds benchmarked to the Dow Jones Industrial Average (DJIA).

- **Active fund:** aims to outperform the DJIA on a risk-adjusted basis. It uses Black–Litterman expected returns and builds a constrained, maximum-Sharpe portfolio of 12 stocks.
- **Passive fund:** aims to track the DJIA closely. It fully replicates the price-weighted index and adds index-futures overlays.
- **Evaluation period:** 1 Oct 2024 – 20 Nov 2025
- **Initial capital:** USD 50m per fund

## 1. Methodology

- **Risk model:** covariance shrinkage with the intensity chosen by cross-validation. It is benchmarked against the sample, constant-correlation and EWMA estimators.
- **Expected returns:** a Black–Litterman model combining the market-implied equilibrium prior with absolute views derived from sell-side target prices.
- **Optimisation:** maximum Sharpe ratio, subject to a 10% single-stock cap, sector bands, and a long-only, fully invested constraint.
- **Rebalancing:** drift-based, triggered when any weight moves more than ±5% from its target.
- **Factor analysis:** Fama–French 3-factor regression (Mkt-RF, SMB, HML), used as a style and risk diagnostic.
- **Market risk:** 95% and 99% Value-at-Risk from a Monte Carlo simulation with 10,000 paths.
- **Stress testing:** the portfolio is replayed across bullish, bearish and stable historical regimes.
- **Passive replication:** price-weighted, including the Nov-2024 index change (INTC and DOW replaced by NVDA and SHW).
- **Derivatives overlay:** DJIA futures used for cash equitization and beta hedging.

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
  - 95%: ending value of USD 40.4m, a loss of about USD 9.6m (19%)
  - 99%: ending value of USD 38.2m, a loss of about USD 11.8m (24%)
- **Scenario analysis (2-year total return):** Bullish +41.45%, Bearish -50.61%, Stable +45.58%
- **Factor exposure:** market risk is the dominant driver. Size exposure is limited because the universe is large-cap, and value exposure is modest.

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

## 3. Skills Demonstrated

- **Portfolio construction:** Bayesian return estimation (Black–Litterman), constrained mean–variance optimisation, drift-band rebalancing.
- **Risk modelling:** shrinkage covariance, Monte Carlo VaR, regime-based stress testing, drawdown and rolling-beta analysis.
- **Performance attribution:** Fama–French factor decomposition, Sharpe, tracking error, Jensen's alpha.
- **Derivatives overlay:** index-futures sizing for cash equitization and beta hedging.
- **Index replication:** price-weighted DJIA tracking through constituent changes.
- **Quant research stack:** end-to-end Python pipeline (pandas, NumPy, SciPy, scikit-learn, yfinance, matplotlib).

## 4. Key Takeaways

- **Active management added value** within a disciplined framework, with a Sharpe ratio of 1.60 against 0.67 for the DJIA.
- **Black–Litterman** gives a structured way to combine market equilibrium with fundamental views.
- **Constraints and shrinkage** turn a mathematical optimisation into an implementable, diversified allocation.
- **Downside risk varies sharply across market regimes**, so a single VaR figure understates tail exposure.
- **Index futures serve two roles:** keeping full benchmark exposure (equitization) or neutralising it (hedging).

## 5. Limitations

- Results depend on the Black–Litterman prior and on the analyst views.
- Correlations are unstable across market regimes.
- The model makes simplified distributional assumptions.
- Futures basis risk and roll costs are not fully modelled.

## Disclaimer

Developed for academic purposes. Nothing in this repository is investment advice. Past and simulated performance does not guarantee future results.
