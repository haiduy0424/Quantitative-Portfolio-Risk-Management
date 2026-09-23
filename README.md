# DJIA Active vs. Passive: A Quantitative Portfolio Study

This project builds two USD 50m long-only funds benchmarked against the DJIA and tests whether a quantitative active strategy can beat the index on a risk-adjusted basis.

- **Active fund:** picks and weights DJIA stocks using Black–Litterman model return forecasts and a constrained optimizer.
- **Passive fund:** replicates the DJIA and uses index futures to manage cash and market exposure.

Both funds are built on data up to Sep 2024 and tested on unseen data from Oct 2024 to Nov 2025.

## 1. Results (out-of-sample)

| Active fund | Portfolio | DJIA | SPY |
|---|---|---|---|
| Total return | **42.1%** | 18.0% | 9.5% |
| Ann. volatility | 20.9% | 18.9% | 16.5% |
| Sharpe ratio | **1.60** | 0.67 | 0.32 |
| Max drawdown | (20.6%) | (18.8%) | (16.4%) |

**Passive fund:** tracking error 0.22% (target < 5%). Futures cash equitization removed cash-drag tracking error (0.82% → ~0).

## 2. Approach

**2.1 Active fund**
- **Covariance:** Estimate the risk model with cross-validated shrinkage to reduce estimation noise.
- **Expected returns:** Blend market-implied equilibrium returns with analyst views via Black–Litterman model instead of relying on historical averages.
- **Optimization:** Maximize the Sharpe ratio under long-only, sector and single-stock limits, keeping the 12 highest-conviction names.
- **Rebalancing:** Rebalance only when a holding drifts more than 5% from its target, net of transaction costs.
- **Risk:** Measure downside risk with Monte Carlo VaR and stress-test the portfolio across bullish, bearish and stable regimes.
- **Attribution:** Decompose returns with a Fama–French factor model to separate market, size and value exposures.

**2.2 Passive fund**
- **Replication:** Hold all DJIA constituents at their price-weighted index weights.
- **Reconstitution:** Rebalance only when index membership changes, such as the Nov 2024 switch from INTC and DOW to NVDA and SHW.
- **Cash equitization:** Use long DJIA futures to put the 5% cash buffer to work and avoid cash drag.
- **Beta hedging:** Test a short-futures overlay that neutralizes market exposure to limit drawdowns.

## 3. Key Findings

- **Risk-adjusted outperformance:** The active fund more than doubled the DJIA's Sharpe ratio (1.60 vs 0.67) with only about 2 pp higher volatility.
- **Return drivers:** Factor regression shows market exposure as the only significant driver, with no meaningful size or value bets.
- **Stock contribution:** GS, JPM, INTC and CSCO contributed most to portfolio return, led by financials and technology.
- **Hidden tail risk:** A 2008-style stress test implies a loss of about 51% over two years, far beyond the realized drawdown.
- **Low turnover:** Weights stayed within the ±5% drift band throughout the period, so no rebalancing costs were incurred.
- **Near-perfect replication:** Futures equitization cut the passive fund's cash-drag tracking error from 0.82% to near zero.

## 4. Key Learnings

- **Portfolio construction:** Applying Bayesian estimation and constrained optimization to build investable portfolios.
- **Risk modeling:** Measuring and stress-testing portfolio risk with robust covariance and simulation methods.
- **Performance attribution:** Evaluating returns through factor models and risk-adjusted metrics.
- **Research discipline:** Designing unbiased out-of-sample backtests and interpreting results critically.
- **Derivatives & implementation:** Using index futures for cash management and hedging.
- **Quantitative programming:** Building an end-to-end research pipeline in Python.

## Disclaimer
This is an academic team project and not an investment advice.
