# DJIA Active vs. Passive: A Quantitative Portfolio Study

**Question:** Can a Bayesian, constraint-aware allocation beat a price-weighted index on a risk-adjusted basis, and how closely can that same index be tracked in practice?

Two USD 50m long-only DJIA funds, calibrated only on data from **Oct 2018 to Sep 2024** and evaluated **out-of-sample from 1 Oct 2024 to 20 Nov 2025**:

- **Active:** Black–Litterman returns + shrinkage covariance → constrained max-Sharpe portfolio
- **Passive:** price-weighted full replication + index-futures overlay

## 1. Results (out-of-sample)

| Active fund | Portfolio | DJIA | SPY |
|---|---|---|---|
| Total return | **42.1%** | 18.0% | 9.5% |
| Ann. volatility | 20.9% | 18.9% | 16.5% |
| Sharpe ratio | **1.60** | 0.67 | 0.32 |
| Max drawdown | −20.6% | −18.8% | −16.4% |

**Passive fund:** tracking error **0.22%** (target < 5%). Futures cash equitization removed cash-drag tracking error (0.82% → ~0).

## 2. Approach

**Active fund**
1. **Covariance:** Ledoit–Wolf shrinkage, with the shrinkage intensity chosen by cross-validated out-of-sample log-likelihood (train ≤ 2022, test 2023–24) ⟶ Raw sample covariance is unstable for 30 assets over 6 years.
2. **Expected returns:** Black–Litterman. The prior is market-cap-implied equilibrium returns, updated with absolute views from sell-side target prices and a view-uncertainty matrix Ω built from confidence levels. The goal is to avoid relying on historical means.
3. **Optimization:** max-Sharpe on the BL posterior. Constraints: long-only, fully invested, sector budgets from a top-down macro view, and a single-name cap. The 12 highest-conviction names are kept.
4. **Rebalancing:** ±5% drift band with 15 bps transaction costs ⟶ Weights never breached the band, so turnover stayed at zero.
5. **Risk:** 10,000-path Monte Carlo on the shrunk covariance with KDE-smoothed marginals gives 1-year VaR of 19% (95%) and 24% (99%). Replaying the portfolio in a bear regime (2007–09) loses 51% over two years, so the portfolio is **not** defensive.
6. **Attribution:** Fama–French 3-factor OLS with HAC errors ⟶ Only the market factor is significant; SMB and HML are not. The excess return comes from a quality large-cap tilt, not from size or value.

**Passive fund**
- Price-weighted replication with event-driven reconstitution (Nov 2024: NVDA and SHW replace INTC and DOW).
- A 5% cash buffer is equitized with long DJIA futures, sized as `N = cash / (index level × multiplier)`.
- A short-futures overlay is tested as a beta-neutral variant for drawdown control.

## 3. Limitations

- Results come from a single 14-month test window, which is too short to prove skill statistically.
- Expected returns rely on sell-side target prices, and view confidences are set by heuristic rather than estimated.
- The Monte Carlo assumes Gaussian dependence, which likely understates joint tail risk.
- Transaction costs are simplified, and futures basis, roll costs and market impact are not modeled.

## 4. Key Learnings

- **Portfolio construction**: Bayesian return estimation and constrained mean-variance optimization.
- **Risk modeling**: covariance shrinkage, Monte Carlo VaR and scenario stress testing.
- **Performance attribution**: factor models, risk-adjusted metrics and tracking error.
- **Research discipline**: out-of-sample testing, avoiding look-ahead bias and separating skill from market exposure.
- **Implementation**: index replication, rule-based rebalancing and futures overlays.

## 5. Repository

```
├── Main.ipynb                           # end-to-end pipeline: data → risk model → BL → optimization → backtest → risk → attribution
├── utils.py                             # covariance, Black–Litterman, backtesting, VaR, factor models, passive/futures modules
├── Report.pdf                           # full fund report
├── target_prices_djia.xlsx              # analyst target prices (BL views)
└── F-F_Research_Data_Factors_daily.xlsx # Fama–French factors
```

```bash
pip install pandas numpy scipy scikit-learn statsmodels PyPortfolioOpt yfinance matplotlib seaborn plotly openpyxl
jupyter notebook Main.ipynb
```

---
## Disclaimer
*This is an cademic team project and not an investment advice.*
