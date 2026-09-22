Active & Passive DJIA Fund Management

A quantitative portfolio management project simulating two USD 50m equity funds benchmarked against the DJIA — one actively managed, one passively replicated — covering the full research pipeline from risk modeling to derivatives overlays.

Overview
Active fund: a constrained, optimization-driven equity portfolio built on Bayesian return forecasting, aiming to outperform the benchmark on a risk-adjusted basis
Passive fund: a full index replication strategy enhanced with futures overlays for cash management and risk exposure control
Evaluation period: Oct 2024 – Nov 2025, USD 50m initial capital per fund
Methodology

Portfolio Construction

Bayesian expected-return estimation blending a market-implied equilibrium prior with subjective, analyst-driven views, rather than relying on historical averages alone
Constrained mean-variance optimization to maximize risk-adjusted return, subject to sector allocation limits, single-stock concentration caps, and a long-only, fully invested mandate
Systematic, rules-based rebalancing triggered by portfolio drift beyond a defined threshold, balancing responsiveness against unnecessary turnover

Risk Management

Robust covariance estimation using shrinkage techniques, with shrinkage intensity calibrated via cross-validation to improve out-of-sample stability over raw sample covariance
Simulation-based tail-risk analysis (Monte Carlo Value-at-Risk) across multiple confidence levels to capture the range of plausible portfolio outcomes
Scenario and stress testing across bullish, bearish and stable market regimes to assess how the portfolio behaves outside of a single historical path

Performance Attribution

Factor-based return decomposition to separate market, size and value exposures and identify the true drivers of performance
Risk-adjusted performance evaluation using standard industry metrics, benchmarked consistently against both the active and passive strategies
Tracking error analysis to quantify how closely the passive fund follows its benchmark, and where deviations originate

Index Replication & Derivatives

Price-weighted full replication of a benchmark index, including correct handling of index composition changes over the evaluation period
Futures overlay strategies for cash equitization, deploying idle cash to minimize performance drag, and for beta hedging, reducing unwanted market exposure
Hedge-ratio sizing and contract-level exposure management to align derivatives positions with the fund's risk objectives
Skills & Competencies Demonstrated
Quantitative portfolio construction and optimization under real-world constraints
Applied Bayesian methods for return forecasting in an asset management context
Statistical risk modeling, including shrinkage estimation and Monte Carlo simulation
Performance attribution and factor analysis for evaluating fund performance
Derivatives-based risk and exposure management
End-to-end quantitative research workflow: data acquisition, modeling, optimization, backtesting and reporting
Strong applied Python skills for building a full quantitative research pipeline from scratch
Tools

Python (pandas, NumPy, SciPy, scikit-learn, statsmodels, PyPortfolioOpt, yfinance), Jupyter Notebook

Disclaimer

This is an academic project. Nothing here constitutes investment advice.
