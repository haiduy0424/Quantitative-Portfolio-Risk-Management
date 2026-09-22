# Active & Passive DJIA Fund Management: Black–Litterman Optimisation, Risk Modelling and Futures Overlays

A quantitative portfolio and risk management study of two simulated USD 50m DJIA-benchmarked equity funds, built as a Python pipeline covering covariance estimation, Bayesian return forecasting, constrained optimisation, tail-risk simulation and futures overlays.

- **Active fund:** a constrained max-Sharpe portfolio of 12 DJIA stocks, built on Black–Litterman expected returns and aiming to beat the index on a risk-adjusted basis.
- **Passive fund:** a price-weighted full replication of the DJIA, with index-futures overlays for cash equitization and beta hedging.
- **Evaluation period:** 1 Oct 2024 – 20 Nov 2025 · **Initial capital:** USD 50m per fund

## 1. Methodology

**Active fund**
- **Risk model:** Ledoit–Wolf shrinkage covariance ⟶ The shrinkage intensity is chosen by cross-validated out-of-sample likelihood.
- **Expected returns:** Black–Litterman model, which updates a market-implied equilibrium prior with absolute views derived from sell-side target prices.
- **Optimisation:** maximum Sharpe ratio, subject to top-down sector allocations, a 10% cap on any single stock, and long-only, fully invested weights.
- **Rebalancing:** drift-based, triggered when any weight moves more than ±5% from its target.
- **Risk analytics:** Fama–French 3-factor attribution, Monte Carlo VaR at 95% and 99% (10,000 simulations), and stress tests across bullish, bearish and stable regimes.

**Passive fund**
- **Replication:** price-weighted, including the Nov-2024 index change (INTC and DOW replaced by NVDA and SHW). 5% of NAV is held in cash to represent execution frictions.
- **Futures overlays:** long DJIA futures for cash equitization and short DJIA futures for beta hedging ⟶ Contracts are sized as *cash ÷ (index level × multiplier)*.

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

- **Portfolio construction:** Black–Litterman return estimation, constrained mean–variance optimisation, drift-band rebalancing.
- **Risk modelling:** shrinkage covariance estimation, Monte Carlo VaR, regime-based stress testing.
- **Attribution:** Fama–French factor regression, Sharpe ratio, tracking error, Jensen's alpha.
- **Index replication and derivatives:** price-weighted index tracking, futures hedge-ratio sizing.
- **Tooling:** Python (pandas, NumPy, SciPy, statsmodels, PyPortfolioOpt, yfinance).

## 4. Key Takeaways

- **The active fund more than doubled the benchmark's risk-adjusted return** (Sharpe 1.60 vs 0.67 for the DJIA) while taking only slightly more volatility and drawdown.
- **Market beta explains most of the return** ⟶ Size and value exposures are not statistically significant, so the fund is a large-cap tilt around a market core rather than a separate factor bet.
- **Tail risk depends on the regime:** +41% in a bullish replay against −51% in a bearish one. A single VaR figure therefore understates drawdown risk.
- **Futures can move exposure either way** ⟶ Long positions restore full benchmark exposure, while a short overlay removes beta at the cost of benchmark returns.

## 5. Limitations

- Allocations are sensitive to the equilibrium prior and to the analyst views.
- The optimisation assumes a stable covariance structure, and the Gaussian dependence in the simulation understates tail dependence.
- Transaction costs, liquidity constraints and futures basis/roll costs are not modelled.
- Results come from a single evaluation window of about 14 months.

## Disclaimer

This is an academic project and nothing in this repository constitutes investment advice.
