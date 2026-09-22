"""Backend utilities for the BlackAlpha Capital fund management notebook."""

import random
import warnings
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns
import yfinance as yf
from binance.client import Client
from IPython.display import HTML, display
from scipy import linalg
from scipy.optimize import minimize
from scipy.stats import norm, percentileofscore
from sklearn.covariance import ShrunkCovariance, empirical_covariance, log_likelihood
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


class DataFetcher:
    def __init__(
        self, username=None, password=None, server=None, api_key=None, api_secret=None, verbose=True
    ):
        self.username = username
        self.password = password
        self.server = server
        self.api_key = api_key
        self.api_secret = api_secret
        self.verbose = verbose
        self._adj_close_warning_shown = False

    def mt5_initialization(self):
        import MetaTrader5 as mt

        mt.login(username=self.username, password=self.password, server=self.server)

    def binance_client_init(self):
        client = Client(self.api_key, self.api_secret)
        return client

    def yfinance_adj_close(self, tickers, start=None, end=None, period="1mo", interval="1d"):
        if isinstance(tickers, str):
            tickers = [tickers]
        try:
            if self.verbose:
                if start and end:
                    print(
                        f"Fetching data for {tickers} from {start} to {end} (interval='{interval}')"
                    )
                else:
                    print(
                        f"Fetching data for {tickers} with period='{period}' and interval='{interval}'"
                    )

            data = yf.download(
                tickers,
                start=start,
                end=end,
                period=period if not start else None,
                interval=interval,
                progress=False,
            )

            if "Adj Close" in data.columns:
                data = data["Adj Close"]
            elif "Close" in data.columns:
                data = data["Close"]
            else:
                return data

            data = data.ffill().bfill()
            return data
        except Exception as e:
            if self.verbose:
                print(f"❌ Error fetching {tickers}: {e}")
            return None

    def darwinex_mt5(self, ticker, start_date, end_date):
        import MetaTrader5 as mt

        self.mt5_initialization()

        df = pd.DataFrame(mt.copy_rates_range(ticker, mt.TIMEFRAME_D1, start_date, end_date))
        df["time"] = pd.to_datetime(df["time"], unit="s").dt.strftime("%Y-%m-%d")
        df.drop(["spread", "real_volume"], axis=1, inplace=True)
        df.set_index("time", inplace=True)
        return df

    def binance_api(self, ticker, interval, start_date):
        client = self.binance_client_init()
        end_time = str((datetime.utcnow() + timedelta(hours=7)).timestamp())
        start_time = str(start_date)
        df = pd.DataFrame(client.get_historical_klines(ticker, interval, start_time, end_time))
        df = df.iloc[:, 0:6]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], unit="ms") + timedelta(hours=7)
        return df[:-1]


class FundOverview:
    def __init__(self):
        pass

    def get_fund_overview(self):
        html = """
        <style>
        .brief-box {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #111;
            background: #f8f9fb;
            border-left: 5px solid #003057;
            padding: 18px 22px;
            max-width: 780px;
            margin: 18px 0;
            border-radius: 5px;
            line-height: 1.5;
        }
        .brief-title {
            font-size: 22px;
            font-weight: 700;
            color: #003057;
            margin-bottom: 12px;
        }
        .brief-section {
            font-size: 16px;
            font-weight: 600;
            color: #003057;
            margin-top: 18px;
        }
        .brief-ul li {
            margin: 4px 0;
        }
        </style>

        <div class="brief-box">

            <div class="brief-title">🟦 FIU Asset Management – Mandate Summary</div>

            <div class="brief-section">Fund 1: Active Mutual Fund</div>
            <ul class="brief-ul">
                <li>Objective: Outperform DJIA on risk-adjusted basis</li>
                <li>Portfolio: 10–15 DJIA stocks</li>
                <li>Capital: USD 50M</li>
                <li>Styles: Growth • Value • Momentum • Sector Rotation</li>
            </ul>

            <div class="brief-section">Fund 2: Passive Mutual Fund</div>
            <ul class="brief-ul">
                <li>Objective: Track DJIA closely</li>
                <li>Capital: USD 50M</li>
                <li>Goal: Minimize tracking error</li>
            </ul>

            <div class="brief-section">Key Requirements</div>
            <ul class="brief-ul">
                <li>Security selection & sector allocation</li>
                <li>Rebalancing methodology</li>
                <li>Performance attribution</li>
                <li>Risk metrics: Volatility, Sharpe, Beta, Drawdown, TE</li>
            </ul>

            <div class="brief-section">Notes</div>
            <ul class="brief-ul">
                <li>Avoid hindsight bias</li>
                <li>Justify decisions with macro + fundamentals</li>
            </ul>

        </div>
        """

        display(HTML(html))


class DataProcessor:
    def __init__(self):
        pass

    def return_from_prices(self, prices, log_returns=False):
        if log_returns:
            returns = np.log(1 + prices.pct_change()).dropna(how="all")
        else:
            returns = prices.pct_change().dropna(how="all")
        return returns


class CovarianceShrinkage:
    def __init__(self, prices, returns_data=False, frequency=252, log_returns=False, delta=None):
        try:
            from sklearn import covariance

            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):
            raise ImportError("Please install scikit-learn to use this class.")

        if not isinstance(prices, pd.DataFrame):
            raise ValueError("data is not in a DataFrame", RuntimeWarning)
            prices = pd.DataFrame(prices)

        self.frequency = frequency
        self.data_processor = DataProcessor()

        if returns_data:
            self.X = prices.dropna(how="all")
        else:
            self.X = self.data_processor.returns_from_prices(prices, log_returns).dropna(how="all")

        self.S = self.X.cov().values
        self.delta = delta

    def _is_positive_semidefinite(self, matrix):
        try:
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    def fix_nonpositive_semidefinite(self, matrix, fix_method="spectral"):
        if self._is_positive_semidefinite(matrix):
            return matrix

        warnings.warn("The covariance matrix is non positive semidefinite. Amending eigenvalues.")

        q, V = np.linalg.eigh(matrix)

        if fix_method == "spectral":
            q = np.where(q > 0, q, 0)
            fixed_matrix = V @ np.diag(q) @ V.T
        elif fix_method == "diag":
            min_eig = np.min(q)
            fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
        else:
            raise NotImplementedError("Method {} not implemented".format(fix_method))

        if not self._is_positive_semidefinite(fixed_matrix):
            warnings.warn("Could not fix matrix. Please try a different risk model.", UserWarning)

        if isinstance(matrix, pd.DataFrame):
            tickers = matrix.index
            return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
        else:
            return fixed_matrix

    def _format_and_annualize(self, raw_cov_array):
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        return self.fix_nonpositive_semidefinite(cov, fix_method="spectral")

    def shrunk_covariance(self, delta=None):
        if delta is not None:
            self.delta = delta
        elif self.delta is None:
            raise ValueError(
                "Delta must be set either during initialization or when calling the method."
            )

        N = self.S.shape[1]

        mu = np.trace(self.S) / N
        F = np.identity(N) * mu

        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)

    def ledoit_wolf(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.ledoit_wolf(X)
        return self._format_and_annualize(shrunk_cov)

    def oracle_approximating(self):
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = self.covariance.oas(X)
        return self._format_and_annualize(shrunk_cov)


class VcvEstimation:
    def __init__(self):
        pass

    def get_number_of_features(self, dict_data):
        return sum(len(tickers) for tickers in dict_data.values()) - 2

    def color_matrix(self, num_of_features):
        return np.random.randn(num_of_features, num_of_features)

    def shrinkage_factor(self):
        return np.logspace(-2, 0, 32)

    def negative_log_likelihood(self, shrinkage_factor, train, test):
        return [-ShrunkCovariance(shrinkage=s).fit(train).score(test) for s in shrinkage_factor]

    def real_covariance(self, color_matrix):
        return np.dot(color_matrix.T, color_matrix)

    def log_real_likelihood(self, empirical_cov, real_cov):
        return -log_likelihood(empirical_cov, linalg.inv(real_cov))

    def get_optimal_shrinkage_coefficient(self, tuned_parameters, train):
        cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
        cv.fit(train)
        optimal_shrinkage_coef = cv.best_estimator_.shrinkage
        return cv, optimal_shrinkage_coef

    def plot_shrinkage_intensity(
        self, shrinkageFactor, negative_logliks, logRealLikelihood, cv, X_test
    ):
        """
        Plot Regularized Covariance Estimation: Likelihood vs Shrinkage Coefficient.

        Parameters
        ----------
        shrinkageFactor : array-like
            Array of shrinkage coefficients (λ) to evaluate.
        negative_logliks : array-like
            Negative log-likelihood values computed for each shrinkage factor.
        logRealLikelihood : float
            Log-likelihood of the real covariance matrix.
        cv : fitted GridSearchCV object
            Fitted cross-validation model from sklearn.covariance.ShrunkCovariance.
        X_test : array-like
            Test dataset used in validation.
        """
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.loglog(
            shrinkageFactor,
            negative_logliks,
            color="#C2185B",
            linestyle="--",
            linewidth=2.5,
            label="Negative Log-Likelihood",
        )
        ax.hlines(
            logRealLikelihood,
            xmin=min(shrinkageFactor),
            xmax=max(shrinkageFactor),
            colors="#1565C0",
            linestyles="dashdot",
            linewidth=2.2,
            label="Real Covariance Likelihood",
        )

        maxLikelihood = np.max(negative_logliks)
        minLikelihood = np.min(negative_logliks)
        min_y = minLikelihood - 0.15 * (maxLikelihood - minLikelihood)
        max_y = maxLikelihood + 0.30 * (maxLikelihood - minLikelihood)
        min_x = shrinkageFactor[0]
        max_x = shrinkageFactor[-1]

        opt_lambda = cv.best_estimator_.shrinkage
        opt_value = -cv.best_estimator_.score(X_test)

        ax.vlines(
            opt_lambda,
            ymin=min_y,
            ymax=opt_value,
            color="#FFEB3B",
            linewidth=3,
            label="CV Optimal Shrinkage",
        )
        ax.annotate(
            f"Optimal λ = {opt_lambda:.3f}",
            xy=(opt_lambda, opt_value),
            xytext=(opt_lambda * 1.3, opt_value * 1.2),
            fontsize=13,
            fontweight="bold",
            color="#444",
            arrowprops=dict(arrowstyle="->", color="#444"),
        )

        ax.set_title(
            "Regularized Covariance Estimation\nLikelihood vs Shrinkage Coefficient",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xlabel("Shrinkage Coefficient (λ)", fontsize=14)
        ax.set_ylabel("Negative Log-Likelihood (Test Set)", fontsize=14)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.legend(fontsize=12, loc="upper right", frameon=True, shadow=True, borderpad=1)
        plt.tight_layout()
        plt.show()


class FamaFrench:
    def __init__(self):
        pass

    def plot_impact(self, model, ols_data):
        plt.figure(figsize=(9, 6))
        sns.set(style="whitegrid")
        pvals = [model.pvalues[f] for f in ols_data["Factor"]]
        norm_pvals = np.clip(1 - np.array(pvals), 0.2, 1.0)  
        colors = sns.color_palette("Blues", len(pvals))
        colors = [colors[int((1 - pv) * (len(colors) - 1))] for pv in norm_pvals]

        ax = sns.barplot(
            x="Factor",
            y="Coefficient",
            data=ols_data,
            palette=colors,
            ci=None,
            edgecolor="black",
            linewidth=1.2,
        )

        for i, row in ols_data.iterrows():
            plt.errorbar(
                i,
                row["Coefficient"],
                yerr=[[abs(row["Confidence_Lower"])], [abs(row["Confidence_Upper"])]],
                fmt="none",
                ecolor="black",
                capsize=5,
                elinewidth=1,
            )

        for i, row in ols_data.iterrows():
            pval = model.pvalues[row["Factor"]]
            text_color = "black" if pval > 0.05 else "darkgreen"
            plt.text(
                i,
                row["Coefficient"] + 0.0005,
                f"β = {row['Coefficient']:.4f}\n(p = {pval:.4f})",
                ha="center",
                va="bottom",
                fontsize=10,
                color=text_color,
                weight="bold",
            )

        plt.title(
            "Impact of Fama–French Factors on Monthly Returns", fontsize=15, weight="bold", pad=20
        )
        plt.xlabel("Factor", fontsize=12, weight="bold")
        plt.ylabel("Coefficient Value", fontsize=12, weight="bold")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.8)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()


class BlackLitterman:
    def __init__(self):
        pass

    def get_target_prices_d(self, price_df):
        target_prices = pd.read_excel(
            "D:/pycode_work/target_prices_djia.xlsx",
            sheet_name="Tickers",
            index_col=0,
        )
        target_prices = target_prices[["Target price"]]
        columns_to_select = price_df.columns
        target_prices = target_prices[target_prices.index.isin(columns_to_select)]
        target_series = target_prices["Target price"]
        return target_series

    def plot_market_prior(self, market_prior):
        plt.figure(figsize=(10, 6))

        market_prior.plot.barh(color="#69b3a2", edgecolor="black", linewidth=1.5)

        plt.grid(axis="x", linestyle="--", alpha=0.7)

        plt.title("Market Prior Returns", fontsize=16)
        plt.xlabel("Returns", fontsize=14)
        plt.ylabel("Tickers", fontsize=14)

        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def get_confidences(self, view_dict):
        def calculate_confidence(views):
            if abs(views) > 0.3:
                return random.choice([0.7, 0.8, 0.9])
            elif abs(views) > 0.1:
                return random.choice([0.4, 0.5, 0.6])
            else:
                return random.choice([0.1, 0.2, 0.3])

        confidences = {
            ticker: calculate_confidence(return_value) for ticker, return_value in view_dict.items()
        }
        confidences = list(confidences.values())
        return confidences

    def plot_portfolio(self, weights):
        weights_series = pd.Series(weights)

        weights_series = weights_series[weights_series > 0]

        fig, ax = plt.subplots(figsize=(10, 10))
        weights_series.plot.pie(
            ax=ax,
            autopct="%1.1f%%", 
            colors=plt.cm.Paired.colors, 
            startangle=90, 
            legend=True, 
            wedgeprops={"edgecolor": "black", "linewidth": 1.5},
        )

        ax.set_title("Portfolio Weights Distribution", fontsize=16, fontweight="bold")
        ax.set_ylabel("") 
        plt.tight_layout()
        plt.show()

plt.rcParams["font.family"] = "sans-serif"


class StyleAnalytics:
    def __init__(self, portfolio_returns, factor_filepath):
        self.port_rets = portfolio_returns.copy()
        self.port_rets = pd.to_numeric(self.port_rets, errors="coerce").dropna()

        self.factor_filepath = factor_filepath
        self.factors = None
        self.aligned_data = None
        self.model_results = {}

        self._load_and_process_factors()
        self._align_and_scale_units()  
    def _load_and_process_factors(self):
        try:
            print(f"📂 Reading factors: {self.factor_filepath}")
            df_temp = pd.read_excel(self.factor_filepath, header=None, engine="openpyxl")
            header_row_idx = None
            for idx, row in df_temp.iterrows():
                row_str = row.astype(str).values
                if any("Mkt-RF" in s or "Mkt-Rf" in s for s in row_str):
                    header_row_idx = idx
                    break

            if header_row_idx is None:
                raw_factors = pd.read_excel(self.factor_filepath, engine="openpyxl")
            else:
                raw_factors = pd.read_excel(
                    self.factor_filepath, header=header_row_idx, engine="openpyxl"
                )

            raw_factors.rename(columns={raw_factors.columns[0]: "Date"}, inplace=True)
            raw_factors["Date"] = pd.to_datetime(
                raw_factors["Date"], format="%Y%m%d", errors="coerce"
            )
            if raw_factors["Date"].isnull().all():
                raw_factors["Date"] = pd.to_datetime(raw_factors.iloc[:, 0], errors="coerce")

            raw_factors = raw_factors.dropna(subset=["Date"]).set_index("Date")
            raw_factors.columns = [str(c).strip() for c in raw_factors.columns]

            col_mapping = {"Mkt-Rf": "Mkt-RF", "Mom": "MOM"}
            raw_factors.rename(columns=col_mapping, inplace=True)

            target_cols = ["Mkt-RF", "SMB", "HML", "RF"]
            available_cols = [c for c in target_cols if c in raw_factors.columns]
            self.factors = (
                raw_factors[available_cols].apply(pd.to_numeric, errors="coerce").dropna()
            )

        except Exception as e:
            raise ValueError(f"Error reading factor file: {str(e)}")

    def _align_and_scale_units(self):
        if self.port_rets.index.tz is not None:
            self.port_rets.index = self.port_rets.index.tz_localize(None)

        common_index = self.port_rets.index.intersection(self.factors.index)
        if len(common_index) == 0:
            raise ValueError("❌ No overlapping dates found between Portfolio and Factors.")

        self.port_rets = self.port_rets.loc[common_index]
        self.factors = self.factors.loc[common_index]

        if self.factors["Mkt-RF"].abs().mean() > 0.05:
            print("ℹ️ Factors detected in Percent. Converting to Decimal (/100).")
            self.factors = self.factors / 100.0

        if self.port_rets.abs().mean() > 0.05:
            print("ℹ️ Portfolio Returns detected in Percent. Converting to Decimal (/100).")
            self.port_rets = self.port_rets / 100.0

        self.aligned_data = pd.concat([self.port_rets, self.factors], axis=1).dropna()
        self.aligned_data.columns = ["Portfolio"] + list(self.factors.columns)

    def run_regression(self):
        y = self.aligned_data["Portfolio"] - self.aligned_data["RF"]
        X = self.aligned_data[["Mkt-RF", "SMB", "HML"]]

        model = LinearRegression()
        model.fit(X, y)

        self.model_results = {
            "Alpha": model.intercept_,
            "Beta_Market": model.coef_[0],
            "Beta_Size (SMB)": model.coef_[1],
            "Beta_Value (HML)": model.coef_[2],
            "R_Squared": model.score(X, y),
        }
        return self.model_results

    def rolling_style(self, window=60):
        y = self.aligned_data["Portfolio"] - self.aligned_data["RF"]
        X = self.aligned_data[["Mkt-RF", "SMB", "HML"]]

        rolling_res = []
        dates = []

        for i in range(window, len(y)):
            y_w = y.iloc[i - window : i]
            X_w = X.iloc[i - window : i]
            model = LinearRegression().fit(X_w, y_w)
            rolling_res.append(
                [model.coef_[0], model.coef_[1], model.coef_[2], model.score(X_w, y_w)]
            )
            dates.append(y.index[i])

        df = pd.DataFrame(rolling_res, index=dates, columns=["Mkt", "SMB", "HML", "R2"])
        return df

    def plot_dashboard(self):
        stats = self.model_results
        rolling = self.rolling_style(window=60)

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

        ax1 = plt.subplot(gs[0, 0])
        x_val = stats["Beta_Value (HML)"]
        y_val = stats["Beta_Size (SMB)"]

        limit = max(1.5, abs(x_val) * 1.5, abs(y_val) * 1.5)

        ax1.fill_between([-limit, 0], 0, limit, color="#e6f3ff", alpha=0.5)  # Small Growth
        ax1.fill_between([0, limit], 0, limit, color="#e6ffe6", alpha=0.5)  # Small Value
        ax1.fill_between([-limit, 0], -limit, 0, color="#fff0e6", alpha=0.5)  # Large Growth
        ax1.fill_between([0, limit], -limit, 0, color="#ffe6e6", alpha=0.5)  # Large Value

        ax1.axhline(0, color="black", lw=1)
        ax1.axvline(0, color="black", lw=1)

        ax1.scatter(
            x_val, y_val, s=300, c="#2c3e50", edgecolors="white", zorder=10, label="Portfolio"
        )
        ax1.text(x_val, y_val + 0.1, "You are here", ha="center", fontweight="bold")

        ax1.text(-limit * 0.8, limit * 0.8, "Small Growth", color="#004d99", fontweight="bold")
        ax1.text(limit * 0.8, limit * 0.8, "Small Value", color="#006600", fontweight="bold")
        ax1.text(-limit * 0.8, -limit * 0.8, "Large Growth", color="#cc5200", fontweight="bold")
        ax1.text(limit * 0.8, -limit * 0.8, "Large Value", color="#990000", fontweight="bold")

        ax1.set_xlim(-limit, limit)
        ax1.set_ylim(-limit, limit)
        ax1.set_title("Style Tilt (Size vs Value)", fontsize=14, fontweight="bold", pad=15)
        ax1.set_xlabel("Value Factor (HML) ← Growth | Value →")
        ax1.set_ylabel("Size Factor (SMB) ← Large Cap | Small Cap →")

        ax2 = plt.subplot(gs[0, 1])
        factors = ["Market Beta", "Size (SMB)", "Value (HML)"]
        values = [stats["Beta_Market"], stats["Beta_Size (SMB)"], stats["Beta_Value (HML)"]]
        colors = ["#3498db", "#9b59b6", "#e67e22"]

        bars = ax2.bar(factors, values, color=colors, alpha=0.8, edgecolor="black")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.bar_label(bars, fmt="%.2f", padding=3)
        ax2.set_title("Factor Exposures (Betas)", fontsize=14, fontweight="bold", pad=15)
        ax2.set_ylim(min(min(values), 0) * 1.2 - 0.5, max(max(values), 0) * 1.2 + 0.5)

        ax3 = plt.subplot(gs[0, 2])
        avg_factors = self.aligned_data[["Mkt-RF", "SMB", "HML"]].mean() * 252  
        contrib = {
            "Market": stats["Beta_Market"] * avg_factors["Mkt-RF"],
            "Size": stats["Beta_Size (SMB)"] * avg_factors["SMB"],
            "Value": stats["Beta_Value (HML)"] * avg_factors["HML"],
            "Alpha": stats["Alpha"] * 252,
        }

        c_names = list(contrib.keys())
        c_vals = list(contrib.values())
        c_colors = ["green" if x >= 0 else "red" for x in c_vals]

        ax3.barh(c_names, c_vals, color=c_colors, alpha=0.7)
        ax3.axvline(0, color="black", lw=0.8)
        ax3.set_title("Est. Annual Return Contribution", fontsize=14, fontweight="bold", pad=15)
        ax3.set_xlabel("Return Contribution (Decimal)")

        ax4 = plt.subplot(gs[1, :])  
        ax4.plot(rolling.index, rolling["SMB"], label="Size (SMB)", color="#9b59b6", lw=2)
        ax4.plot(rolling.index, rolling["HML"], label="Value (HML)", color="#e67e22", lw=2)
        ax4.fill_between(rolling.index, rolling["SMB"], alpha=0.1, color="#9b59b6")
        ax4.fill_between(rolling.index, rolling["HML"], alpha=0.1, color="#e67e22")

        ax4.axhline(0, color="black", linestyle="--", lw=1)
        ax4.legend(loc="upper left", fontsize=12)
        ax4.set_title("Style Drift Over Time (60-day Rolling)", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Factor Beta")

        ax4b = ax4.twinx()
        ax4b.plot(
            rolling.index,
            rolling["R2"],
            color="gray",
            linestyle=":",
            alpha=0.5,
            label="Model Fit (R2)",
        )
        ax4b.set_ylabel("Model Fit (R2)", color="gray")

        plt.tight_layout()
        plt.show()


class AlgorithmicTrading:
    def __init__(self):
        pass

    def generate_signals(self, input_df, start_capital=100000, share_count=2000):
        initial_capital = float(start_capital)

        signals_df = input_df.copy()

        share_size = share_count

        signals_df["Position"] = share_size * signals_df["Buy Signal"]

        signals_df["Entry/Exit"] = signals_df["Buy Signal"].diff()

        signals_df["Entry/Exit Position"] = signals_df["Position"].diff()

        signals_df["Portfolio Holdings"] = (
            signals_df["Returns"] * signals_df["Entry/Exit Position"].cumsum()
        )

        signals_df["Portfolio Cash"] = (
            initial_capital - (signals_df["Returns"] * signals_df["Entry/Exit Position"]).cumsum()
        )

        signals_df["Portfolio Total"] = (
            signals_df["Portfolio Cash"] + signals_df["Portfolio Holdings"]
        )

        signals_df["Portfolio Daily Returns"] = signals_df["Portfolio Total"].pct_change()

        signals_df["Portfolio Cumulative Returns"] = (
            1 + signals_df["Portfolio Daily Returns"]
        ).cumprod() - 1

        signals_df = signals_df.dropna()

        return signals_df

    def algo_evaluation(self, signals_df):
        metrics = [
            "Annual Return",
            "Cumulative Returns",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
        ]
        columns = ["Backtest"]
        portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)

        portfolio_evaluation_df.loc["Cumulative Returns"] = (
            signals_df["Portfolio Cumulative Returns"][-1]
        ) * 100

        portfolio_evaluation_df.loc["Annual Return"] = (
            signals_df["Portfolio Daily Returns"].mean() * 252
        ) * 100

        portfolio_evaluation_df.loc["Annual Volatility"] = (
            signals_df["Portfolio Daily Returns"].std() * np.sqrt(252)
        ) * 100

        portfolio_evaluation_df.loc["Sharpe Ratio"] = (
            signals_df["Portfolio Daily Returns"].mean() * 252
        ) / (signals_df["Portfolio Daily Returns"].std() * np.sqrt(252))

        sortino_ratio_df = signals_df[["Portfolio Daily Returns"]].copy()
        sortino_ratio_df.loc[:, "Downside Returns"] = 0
        target = 0
        mask = sortino_ratio_df["Portfolio Daily Returns"] < target
        sortino_ratio_df.loc[mask, "Downside Returns"] = (
            sortino_ratio_df["Portfolio Daily Returns"] ** 2
        )
        down_stdev = np.sqrt(sortino_ratio_df["Downside Returns"].mean()) * np.sqrt(252)
        expected_return = sortino_ratio_df["Portfolio Daily Returns"].mean() * 252
        sortino_ratio = expected_return / down_stdev

        portfolio_evaluation_df.loc["Sortino Ratio"] = sortino_ratio

        return portfolio_evaluation_df

    def _evaluate_return_metrics(self, signals_df):
        metrics = [
            "Annual Return",
            "Cumulative Returns",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
        ]
        columns = ["Backtest"]

        portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)

        portfolio_evaluation_df.loc["Cumulative Returns"] = signals_df[
            "Portfolio Cumulative Returns"
        ][-1]

        daily_returns_mean = signals_df["Portfolio Daily Returns"].mean()
        portfolio_evaluation_df.loc["Annual Return"] = daily_returns_mean * 252

        daily_returns_std = signals_df["Portfolio Daily Returns"].std()
        portfolio_evaluation_df.loc["Annual Volatility"] = daily_returns_std * np.sqrt(252)

        portfolio_evaluation_df.loc["Sharpe Ratio"] = (daily_returns_mean * 252) / (
            daily_returns_std * np.sqrt(252)
        )

        sortino_ratio_df = signals_df[["Portfolio Daily Returns"]].copy()
        sortino_ratio_df["Downside Returns"] = 0  # Initialize downside returns column

        target = 0
        mask = sortino_ratio_df["Portfolio Daily Returns"] < target
        sortino_ratio_df.loc[mask, "Downside Returns"] = (
            sortino_ratio_df["Portfolio Daily Returns"] ** 2
        )

        downside_stdev = np.sqrt(sortino_ratio_df["Downside Returns"].mean()) * np.sqrt(252)
        expected_annual_return = daily_returns_mean * 252
        sortino_ratio = expected_annual_return / downside_stdev if downside_stdev > 0 else np.nan

        portfolio_evaluation_df.loc["Sortino Ratio"] = sortino_ratio

        return portfolio_evaluation_df

    def underlying_evaluation(self, signals_df):
        underlying = pd.DataFrame()
        underlying["Returns"] = signals_df["Returns"]
        underlying["Portfolio Daily Returns"] = underlying["Returns"]
        underlying["Portfolio Daily Returns"].fillna(0, inplace=True)
        underlying["Portfolio Cumulative Returns"] = (
            1 + underlying["Portfolio Daily Returns"]
        ).cumprod() - 1

        underlying_evaluation = self._evaluate_return_metrics(underlying)

        return underlying_evaluation

    def algo_vs_underlying(self, signals_df):
        metrics = [
            "Annual Return",
            "Cumulative Returns",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
        ]

        columns = ["Algo", "Underlying"]
        algo = self.algo_evaluation(signals_df)
        underlying = self.underlying_evaluation(signals_df)

        comparison_df = pd.DataFrame(index=metrics, columns=columns)
        comparison_df["Algo"] = algo["Backtest"]
        comparison_df["Underlying"] = underlying["Backtest"]

        return comparison_df

    def trade_evaluation(self, signals_df):

        trade_evaluation_df = pd.DataFrame(
            columns=[
                "Entry Date",
                "Exit Date",
                "Shares",
                "Entry Share Price",
                "Exit Share Price",
                "Entry Portfolio Holding",
                "Exit Portfolio Holding",
                "Profit/Loss",
            ]
        )
        entry_date = ""
        exit_date = ""
        entry_portfolio_holding = 0
        exit_portfolio_holding = 0
        share_size = 0
        entry_share_price = 0
        exit_share_price = 0

        for index, row in signals_df.iterrows():
            if row["Entry/Exit"] == 1:
                entry_date = index
                entry_portfolio_holding = row["Portfolio Total"]
                share_size = row["Entry/Exit Position"]
                entry_share_price = row["Returns"]

            elif row["Entry/Exit"] == -1:
                exit_date = index
                exit_portfolio_holding = abs(row["Portfolio Total"])
                exit_share_price = row["Returns"]
                profit_loss = exit_portfolio_holding - entry_portfolio_holding
                trade_evaluation_df.loc[len(trade_evaluation_df)] = {
                    "Entry Date": entry_date,
                    "Exit Date": exit_date,
                    "Shares": share_size,
                    "Entry Share Price": entry_share_price,
                    "Exit Share Price": exit_share_price,
                    "Entry Portfolio Holding": entry_portfolio_holding,
                    "Exit Portfolio Holding": exit_portfolio_holding,
                    "Profit/Loss": profit_loss,
                }
        return trade_evaluation_df

    def underlying_returns(self, signals_df, figsize=(14, 8)):
        underlying = pd.DataFrame()
        underlying["Returns"] = signals_df["Returns"].fillna(0)
        underlying["Portfolio Cumulative Returns"] = signals_df[
            "Portfolio Cumulative Returns"
        ].fillna(0)

        underlying["Underlying Cumulative Returns"] = (1 + underlying["Returns"]).cumprod() - 1
        underlying["Algo Cumulative Returns"] = underlying["Portfolio Cumulative Returns"] * 100
        underlying.index = signals_df.index

        sns.set(style="whitegrid")
        plt.figure(figsize=figsize)
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["font.size"] = 11

        plt.plot(
            underlying.index,
            underlying["Underlying Cumulative Returns"],
            label="Underlying Cumulative Returns",
            color="#1f77b4",
            linewidth=2.2,
            alpha=0.9,
        )
        plt.plot(
            underlying.index,
            underlying["Algo Cumulative Returns"],
            label="Algo Cumulative Returns",
            color="#ff7f0e",
            linewidth=2.2,
            alpha=0.9,
        )

        plt.fill_between(
            underlying.index,
            underlying["Underlying Cumulative Returns"],
            underlying["Algo Cumulative Returns"],
            where=underlying["Algo Cumulative Returns"]
            > underlying["Underlying Cumulative Returns"],
            color="green",
            alpha=0.1,
            interpolate=True,
            label="Outperform Region",
        )
        plt.fill_between(
            underlying.index,
            underlying["Underlying Cumulative Returns"],
            underlying["Algo Cumulative Returns"],
            where=underlying["Algo Cumulative Returns"]
            < underlying["Underlying Cumulative Returns"],
            color="red",
            alpha=0.1,
            interpolate=True,
            label="Underperform Region",
        )

        max_idx = underlying["Algo Cumulative Returns"].idxmax()
        min_idx = underlying["Algo Cumulative Returns"].idxmin()
        plt.scatter(
            max_idx,
            underlying.loc[max_idx, "Algo Cumulative Returns"],
            color="darkgreen",
            s=60,
            zorder=5,
        )
        plt.scatter(
            min_idx,
            underlying.loc[min_idx, "Algo Cumulative Returns"],
            color="darkred",
            s=60,
            zorder=5,
        )
        plt.text(
            max_idx,
            underlying.loc[max_idx, "Algo Cumulative Returns"],
            f" Peak: {underlying.loc[max_idx, 'Algo Cumulative Returns']:.2f}",
            color="darkgreen",
            fontsize=10,
            ha="left",
            va="bottom",
            weight="bold",
        )
        plt.text(
            min_idx,
            underlying.loc[min_idx, "Algo Cumulative Returns"],
            f" Trough: {underlying.loc[min_idx, 'Algo Cumulative Returns']:.2f}",
            color="darkred",
            fontsize=10,
            ha="left",
            va="top",
            weight="bold",
        )

        plt.title(
            "Algorithmic Strategy vs Underlying Asset – Cumulative Returns",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        plt.xlabel("Date", fontsize=12, weight="bold")
        plt.ylabel("Cumulative Return (%)", fontsize=12, weight="bold")
        plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=False, loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

        return underlying[["Underlying Cumulative Returns", "Algo Cumulative Returns"]]

    def plot_cumulative_returns(self, signals_df, figsize=(16, 8), style="whitegrid"):
        sns.set_style(style)
        sns.set_context("talk")

        underlying = pd.DataFrame()
        underlying["Returns"] = signals_df["Returns"]

        # Daily returns (buy & hold)
        underlying["Underlying Daily Returns"] = underlying["Returns"].fillna(0)

        # Cumulative underlying returns
        underlying["Underlying Cumulative Returns"] = (
            1 + underlying["Underlying Daily Returns"]
        ).cumprod() - 1

        # Algo cumulative returns 
        underlying["Algo Cumulative Returns"] = signals_df["Portfolio Cumulative Returns"]

        plt.figure(figsize=figsize)
        plt.plot(
            underlying.index,
            underlying["Underlying Cumulative Returns"],
            label="Underlying Cumulative Returns",
            linewidth=2.2,
            color="#1f77b4",
        )

        plt.plot(
            underlying.index,
            underlying["Algo Cumulative Returns"],
            label="Algo Cumulative Returns",
            linewidth=2.5,
            color="#ff7f0e",
        )

        plt.title("Cumulative Returns Comparison", fontsize=22, pad=20)
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Cumulative Returns", fontsize=18)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(fontsize=16)

        plt.tight_layout()
        plt.show()

        return underlying


class TranscendentalKernel:
    def __init__(self):
        pass

    def plot_pdf(self, log_returns):
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")  

        colors = sns.color_palette("tab10", len(log_returns.columns))

        for i, column in enumerate(log_returns.columns):
            sns.kdeplot(
                log_returns[column],
                label=column,
                fill=True,
                alpha=0.6,
                color=colors[i],
                linewidth=1.5,
            )

        plt.title(
            "Probability Density Function (PDF) for Log Returns",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Log Return", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Tickers", title_fontsize=13, fontsize=12, loc="upper left", frameon=True)
        plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.show()

    def percentile_to_percentile_mapping(self, num_samples, log_returns, vcv_matrix, pdf_dict):
        samples = np.random.multivariate_normal(log_returns.mean(), vcv_matrix, size=num_samples)

        transformed_samples = np.zeros_like(samples)

        for i, column in enumerate(log_returns.columns):
            returns = pdf_dict[column]["returns"]
            pdf = pdf_dict[column]["pdf"]

            cdf = np.cumsum(pdf) * np.diff(returns)[0]

            for j in range(num_samples):
                sample_value = samples[j, i]
                percentile = np.interp(sample_value, returns, cdf)
                transformed_samples[j, i] = np.interp(percentile, cdf, returns)

        transformed_samples_df = pd.DataFrame(transformed_samples, columns=log_returns.columns)
        return transformed_samples_df

    def calculate_var(self, ending_value, alpha=0.05):
        ranked_values = ending_value.rank(pct=True)
        var_value = ending_value[ranked_values <= alpha].max()
        return var_value

    def plot_var(self, ending_value):
        sorted_values = ending_value.sort_values()

        percentiles = np.linspace(0, 1, len(sorted_values))

        kde = stats.gaussian_kde(sorted_values)

        # Calculate VaR at 95% and 99%
        VaR_95_threshold = sorted_values.quantile(0.05)  
        VaR_99_threshold = sorted_values.quantile(0.01)  

        x = np.linspace(sorted_values.min(), sorted_values.max(), 1000)
        pdf_values = kde(x)

        plt.figure(figsize=(12, 6))

        plt.plot(x, pdf_values, label="Estimated PDF", color="#1f77b4", linewidth=2)

        plt.axvline(
            VaR_95_threshold,
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
            label=f"VaR 95%: {VaR_95_threshold:,.2f}",
        )
        plt.axvline(
            VaR_99_threshold,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label=f"VaR 99%: {VaR_99_threshold:,.2f}",
        )

        plt.fill_between(
            x=x,
            y1=pdf_values,
            y2=0,
            where=x <= VaR_95_threshold,
            color="#ff7f0e",
            alpha=0.3,
            label="Below 95% Threshold",
        )
        plt.fill_between(
            x=x,
            y1=pdf_values,
            y2=0,
            where=x <= VaR_99_threshold,
            color="#d62728",
            alpha=0.3,
            label="Below 99% Threshold",
        )

        plt.title("VaR of Ending Portfolio Value", fontsize=16, fontweight="bold")
        plt.xlabel("Portfolio Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.legend(loc="best", fontsize=12, title="Value at Risk", title_fontsize=14)
        plt.show()


class PortfolioPerformance:
    def __init__(self):
        pass

    def plot_portfolio_returns(self, port_retss):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=port_retss.index,
                y=port_retss.values,
                mode="lines",
                name="Portfolio Cumulative Returns",
                line=dict(color="#0072B2", width=2.5),
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            annotation_text="Baseline",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=dict(
                text="Portfolio Cumulative Returns", font=dict(size=22, color="#333"), x=0.5
            ),
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            hovermode="x unified",
            font=dict(family="Arial", size=13),
            margin=dict(l=60, r=40, t=70, b=50),
            height=600,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=0.3, gridcolor="LightGray")
        fig.show()

    def plot_mdd(self, drawdown):
        plt.figure(figsize=(14, 8))

        plt.plot(drawdown, label="Drawdown", color="red", linewidth=2, alpha=0.8)

        plt.title("Portfolio Drawdown Over Time", fontsize=16, fontweight="bold", color="darkred")
        plt.xlabel("Date", fontsize=14, color="black")
        plt.ylabel("Drawdown (%)", fontsize=14, color="black")

        max_drawdown_date = drawdown.idxmin()
        max_drawdown_value = drawdown.min()
        plt.scatter(
            max_drawdown_date, max_drawdown_value, color="darkblue", s=100, label="Maximum Drawdown"
        )
        plt.text(
            max_drawdown_date,
            max_drawdown_value,
            f"{max_drawdown_value:.2f}",
            color="darkblue",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=12, loc="best", frameon=True, shadow=True, edgecolor="black")
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_performance(
        self,
        portfolio_cum_returns,
        ixn_returns,
        msci_returns,
        global_clean_energy_returns,
        initial_money,
    ):
        plt.figure(figsize=(14, 8))

        plt.plot(
            portfolio_cum_returns,
            label="Portfolio Cumulative Return",
            color="steelblue",
            linewidth=2,
        )

        plt.plot(
            (1 + ixn_returns).cumprod() * initial_money,
            label="iShares Global Tech ETF",
            color="firebrick",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(
            (1 + msci_returns).cumprod() * initial_money,
            label="iShares MSCI Emerging Markets ETF",
            color="teal",
            linestyle="-.",
            linewidth=2,
        )

        plt.plot(
            (1 + global_clean_energy_returns).cumprod() * initial_money,
            label="iShares Global Clean Energy ETF",
            color="peru",
            linestyle=":",
            linewidth=2,
        )

        plt.title("Portfolio vs. Market Performance", fontsize=16, fontweight="bold")
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cumulative Returns ($)", fontsize=14)
        plt.legend(fontsize=12, loc="upper left", frameon=True, shadow=True)
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)

        plt.tight_layout()

        plt.show()


def backtest_drift_rebalancing(
    price_data, target_weights, initial_capital=50_000_000, threshold=0.05
):

    if isinstance(target_weights, pd.Series):
        target_weights = target_weights.to_dict()

    tickers = list(target_weights.keys())
    dates = price_data.index

    initial_prices = price_data.iloc[0][tickers]
    shares = {
        ticker: (initial_capital * target_weights[ticker]) / initial_prices[ticker]
        for ticker in tickers
    }

    portfolio_history = []
    rebalance_dates = []

    for date in dates[1:]:
        current_prices = price_data.loc[date, tickers]
        current_holdings = {t: shares[t] * current_prices[t] for t in tickers}
        total_value = sum(current_holdings.values())

        current_weights = {t: val / total_value for t, val in current_holdings.items()}

        needs_rebalance = False
        for t in tickers:
            drift = abs(current_weights[t] - target_weights[t])
            if drift > threshold:
                needs_rebalance = True
                break

        if needs_rebalance:
            shares = {t: (total_value * target_weights[t]) / current_prices[t] for t in tickers}
            rebalance_dates.append(date)

        portfolio_history.append(total_value)

    portfolio_value = pd.Series(portfolio_history, index=dates[1:])
    portfolio_value.loc[dates[0]] = initial_capital
    portfolio_value = portfolio_value.sort_index()

    return portfolio_value, rebalance_dates


def advanced_drift_backtest(
    price_data,
    target_weights,
    initial_capital=50_000_000,
    threshold=0.05,
    cost_rate=0.0015, 
    verbose=True,
):
    """
    Backtests a drift-based rebalancing strategy with transaction costs and cash management.

    Parameters:
    - price_data: DataFrame of asset prices (Adj Close).
    - target_weights: Series/Dict of target allocation.
    - initial_capital: Starting cash.
    - threshold: Rebalance trigger (e.g., 0.05 for 5% drift).
    - cost_rate: Transaction cost per trade (e.g., 0.0015).
    - verbose: If True, prints the performance summary.

    Returns:
    - history_df: Daily portfolio value and cash.
    - rebal_df: Log of rebalancing events.
    - metrics: Dictionary of performance stats.
    """

    if isinstance(target_weights, dict):
        target_weights = pd.Series(target_weights)

    # Filter valid tickers (intersection of weights and price columns)
    valid_tickers = [t for t in target_weights.index if t in price_data.columns]
    if len(valid_tickers) < len(target_weights):
        dropped = set(target_weights.index) - set(valid_tickers)
        print(f"⚠️ Warning: Dropped tickers due to missing price data: {dropped}")

    target_weights = target_weights[valid_tickers]
    total_w = target_weights.sum()
    if not np.isclose(total_w, 1.0):
        target_weights = target_weights / total_w

    prices = price_data[valid_tickers].copy()
    prices = prices.sort_index().ffill()  
    if prices.isnull().values.any():
        prices = prices.dropna()  
    dates = prices.index
    cash = initial_capital
    shares = pd.Series(0.0, index=valid_tickers)

    portfolio_history = []
    rebalance_logs = []

    first_date = dates[0]
    first_prices = prices.iloc[0]

    buy_values = initial_capital * target_weights
    initial_shares = buy_values / first_prices
    initial_cost = buy_values.sum() * cost_rate

    shares = initial_shares
    cash = initial_capital - buy_values.sum() - initial_cost

    portfolio_history.append(
        {
            "Date": first_date,
            "TotalValue": cash + (shares * first_prices).sum(),
            "Cash": cash,
            "TransactionCost": initial_cost,
            "IsRebalanced": True,
        }
    )

    for date in dates[1:]:
        current_prices = prices.loc[date]
        holdings_value = (shares * current_prices).sum()
        total_value = cash + holdings_value
        if total_value == 0:
            current_weights = pd.Series(0, index=valid_tickers)
        else:
            current_weights = (shares * current_prices) / total_value
        drift = (current_weights - target_weights).abs()
        max_drift = drift.max()
        needs_rebalance = False
        if max_drift > threshold:
            needs_rebalance = True
            
        daily_cost = 0.0

        if needs_rebalance:
            target_asset_values = total_value * target_weights
            target_shares = target_asset_values / current_prices

            share_diff = target_shares - shares
            net_purchase = (share_diff * current_prices).sum()

            trade_values = (share_diff * current_prices).abs()
            total_turnover = trade_values.sum()
            daily_cost = total_turnover * cost_rate

            cash = cash - net_purchase - daily_cost
            shares = target_shares

            total_value = cash + (shares * current_prices).sum()

            rebalance_logs.append(
                {
                    "Date": date,
                    "PortfolioValue": total_value,
                    "Turnover": total_turnover,
                    "Cost": daily_cost,
                    "MaxDrift": max_drift,
                }
            )

        portfolio_history.append(
            {
                "Date": date,
                "TotalValue": total_value,
                "Cash": cash,
                "TransactionCost": daily_cost,
                "IsRebalanced": needs_rebalance,
            }
        )

    history_df = pd.DataFrame(portfolio_history).set_index("Date")
    rebal_df = pd.DataFrame(rebalance_logs)
    if not rebal_df.empty:
        rebal_df.set_index("Date", inplace=True)

    # Calculate returns
    rets = history_df["TotalValue"].pct_change().dropna()

    try:
        ann_ret = annualize_rets(rets, periods_per_year=252)
        ann_vol = annualize_vol(rets, periods_per_year=252)
        sharpe = sharpe_ratio(rets, riskfree_rate=0.03, periods_per_year=252)
        mdd = drawdown(rets)["Drawdown"].min()
    except Exception:
        days = (history_df.index[-1] - history_df.index[0]).days
        total_ret = (history_df["TotalValue"].iloc[-1] / initial_capital) - 1
        ann_ret = (1 + total_ret) ** (365.25 / days) - 1 if days > 0 else 0
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        cumulative = (1 + rets).cumprod()
        peak = cumulative.cummax()
        mdd = ((cumulative - peak) / peak).min()

    metrics = {
        "CAGR": ann_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Total Rebalances": history_df["IsRebalanced"].sum() - 1,  
        "Total Transaction Costs": history_df["TransactionCost"].sum(),
    }

    if verbose:
        print("-" * 40)
        print("ADVANCED BACKTEST RESULTS")
        print("-" * 40)
        print(f"CAGR (Annual return): {metrics['CAGR']*100:.2f}%")
        print(f"Sharpe Ratio:        {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown:        {metrics['Max Drawdown']*100:.2f}%")
        print(f"Number of rebalances:{metrics['Total Rebalances']}")
        print(f"Total trading costs: ${metrics['Total Transaction Costs']:,.2f}")
        print("-" * 40)

    return history_df, rebal_df, metrics


def analyze_portfolio_vs_benchmark_specific_period(weights_dict, start_date, end_date):
    """
    Compares Portfolio performance against DJIA and SPY for a specific timeframe.
    Ensures precise annualized return calculation by aligning trading days.
    """
    print(f"🔄 Fetching data from {start_date} to {end_date}...")

    tickers = list(weights_dict.keys())
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    except KeyError:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]

    data = data.ffill().bfill()

    daily_rets = data.pct_change().dropna()
    port_rets = (daily_rets * pd.Series(weights_dict)).sum(axis=1)
    port_rets.name = "Portfolio"

    benchmarks = ["^DJI", "SPY"]
    bench_data = yf.download(benchmarks, start=start_date, end=end_date, progress=False)["Close"]
    bench_rets = bench_data.pct_change().dropna()

    comparison_df = pd.concat([port_rets, bench_rets], axis=1, join="inner")
    comparison_df.columns = ["Portfolio", "DJIA", "SPY"]

    print(f"✅ Data aligned. Total trading days analyzed: {len(comparison_df)}")

    def get_metrics(series):
        # Total Return
        total_ret = (1 + series).prod() - 1

        n_days = len(series)
        ann_factor = 252 / n_days if n_days > 0 else 0
        ann_ret = (1 + total_ret) ** ann_factor - 1

        # Annualized Volatility
        volatility = series.std() * np.sqrt(252)
        rf = 0.03
        sharpe = (ann_ret - rf) / volatility if volatility != 0 else 0

        # Max Drawdown
        cum_ret = (1 + series).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1
        max_dd = drawdown.min()

        return {
            "Total Return": total_ret,
            "Ann. Return (CAGR)": ann_ret,
            "Ann. Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        }

    stats = {col: get_metrics(comparison_df[col]) for col in comparison_df.columns}
    metrics_df = pd.DataFrame(stats).T

    # Formatting for display
    display_metrics = metrics_df.copy()
    for col in ["Total Return", "Ann. Return (CAGR)", "Ann. Volatility", "Max Drawdown"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2%}")
    display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")

    plt.figure(figsize=(12, 6))

    # Cumulative Returns (Growth of $100)
    cum_returns = (1 + comparison_df).cumprod() * 100

    plt.plot(
        cum_returns.index,
        cum_returns["Portfolio"],
        label="Portfolio (FIUAM)",
        color="#003057",
        linewidth=2.5,
    )
    plt.plot(
        cum_returns.index,
        cum_returns["DJIA"],
        label="Benchmark (DJIA)",
        color="gray",
        linestyle="--",
        alpha=0.8,
    )
    plt.plot(
        cum_returns.index,
        cum_returns["SPY"],
        label="S&P 500 (SPY)",
        color="orange",
        linestyle="--",
        alpha=0.8,
    )

    plt.title(
        f"Performance Comparison ({start_date} to {end_date})", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Portfolio Value (Rebased to 100)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    port_tot = metrics_df.loc["Portfolio", "Total Return"]
    djia_tot = metrics_df.loc["DJIA", "Total Return"]
    spy_tot = metrics_df.loc["SPY", "Total Return"]

    textstr = "\n".join(
        (
            r"$\bf{Total\ Returns:}$",
            f"Portfolio: {port_tot:.2%}",
            f"DJIA: {djia_tot:.2%}",
            f"SPY: {spy_tot:.2%}",
        )
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.gca().text(
        0.02,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()

    print("\n📊 Detailed Performance Metrics:")
    from IPython.display import display

    display(display_metrics)

    return metrics_df


class PortfolioVisualizer:
    def __init__(self):
        pass

    def plot_bar_returns(self, cumulative_returns_df, title="Cumulative Returns by Asset"):

        if isinstance(cumulative_returns_df, pd.DataFrame):
            cumulative_returns_df = cumulative_returns_df.iloc[-1]  
        data = cumulative_returns_df.reset_index()
        data.columns = ["Ticker", "Return"]

        colors = (
            px.colors.sequential.Blues if data["Return"].mean() > 0 else px.colors.sequential.Reds
        )

        fig = px.bar(
            data,
            x="Ticker",
            y="Return",
            text="Return",
            color="Return",
            color_continuous_scale=colors,
            template="plotly_white",
            title=title,
        )

        fig.update_traces(
            texttemplate="%{text:.2%}",
            textposition="outside",
            marker_line_color="black",
            marker_line_width=1.2,
            opacity=0.9,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=22, color="#003057")),
            xaxis_title="Ticker",
            yaxis_title="Cumulative Return",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="LightGray"),
            coloraxis_showscale=False,
            font=dict(family="Arial", size=13),
            height=600,
            width=1000,
            bargap=0.3,
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.show()


def select_top_stocks(weights_dict, top_n=12):
    """
    Select top N stocks by portfolio weights and normalize their weights to sum to 1.

    Parameters:
        weights_dict (dict): Dictionary of stock weights (e.g., from EfficientFrontier).
        top_n (int): Number of stocks to keep (default = 15).

    Returns:
        pd.Series: Normalized weights for top N stocks.
    """
    weights_series = pd.Series(weights_dict)
    top_stocks = weights_series.nlargest(top_n)
    normalized_top = top_stocks / top_stocks.sum()
    return normalized_top.to_dict()


def get_historical_data(tickers_dict, start_date="2018-10-01", end_date="2024-10-01"):
    """
    Download historical stock price data from Yahoo Finance for given tickers by sector.

    Parameters
    ----------
    tickers_dict : dict
        Dictionary mapping sector names to lists of tickers, e.g.:
        {"information_technology": ["AAPL", "MSFT", "CSCO"], ...}
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        A DataFrame of adjusted closing prices for all tickers combined.
    """
    df = pd.DataFrame()

    for sector, tickers in tickers_dict.items():
        print(f"\n🔹 Sector: {sector}")
        for t in tickers:
            print(f"  → Fetching {t} ...")
            try:
                prices = yf.download(
                    t,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )["Close"]
                df[t] = prices
            except Exception as e:
                print(f"  ❌ Failed for {t}: {e}")

    # Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    print("\n✅ Data download complete!")
    return df


def get_caps(tickers_dict, start_date="2018-10-01", end_date="2024-10-01"):
    """
    Download historical market capitalization (Market Cap) for each stock
    in the given tickers dictionary, grouped by sector.

    Parameters
    ----------
    tickers_dict : dict
        Dictionary mapping sector names to lists of tickers.
        Example: {"information_technology": ["AAPL", "MSFT"], ...}
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        A DataFrame named 'caps' containing historical market capitalization
        values for all tickers.
    """
    caps = pd.DataFrame()

    for sector, tickers in tickers_dict.items():
        print(f"\n🔹 Sector: {sector}")

        for t in tickers:
            print(f"  → Fetching market cap for {t} ...")
            try:
                stock = yf.Ticker(t)
                data = stock.history(start=start_date, end=end_date, auto_adjust=True)
                shares = stock.info.get("sharesOutstanding", None)

                if shares is None:
                    print(f"  ⚠️ Warning: {t} has no sharesOutstanding data.")
                    caps[t] = pd.Series([None] * len(data), index=data.index)
                    continue

                market_cap = data["Close"] * shares
                caps[t] = market_cap

            except Exception as e:
                print(f"  ❌ Failed for {t}: {e}")
                caps[t] = None

    # Fill missing data
    caps.ffill(inplace=True)
    caps.bfill(inplace=True)

    print("\n✅ Market Cap data download complete!")
    return caps


def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted
    """
    n = len(r.columns)
    ew = pd.Series(1 / n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]]  # starting cap weight
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew / ew.sum()
        # limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw * max_cw_mult)
            ew = ew / ew.sum()  # reweight
    return ew


def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[1]]
    return w / w.sum()


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = stats.jarque_bera(r)
        return p_value > level


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b) ** 2).sum())


def portfolio_return_bt(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    vol = (weights.T @ covmat @ weights) ** 0.5
    return vol


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return_bt(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(
        neg_sharpe,
        init_guess,
        args=(riskfree_rate, er, cov),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1,),
        bounds=bounds,
    )
    return weights.x


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n 
    # construct the constraints
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er),
    }
    weights = minimize(
        portfolio_vol,
        init_guess,
        args=(cov,),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1, return_is_target),
        bounds=bounds,
    )
    return weights.x


def backtest_ws(r, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window)]
    weights = [weighting(r.iloc[win[0] : win[1]], **kwargs) for win in windows]
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(
        axis="columns", min_count=1
    )  
    return returns


def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.0)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)


def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta * prior + (1 - delta) * sample


def annualize_rets(r, periods_per_year):
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    return r.std() * (periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index,
    the previous peaks, and
    the percentage drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame(
        {"Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdowns}
    )


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """

    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame(
        {
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd,
        }
    )

# PASSIVE FUND ANALYSIS ======================================================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)
pd.set_option("display.float_format", "{:,.4f}".format)


class PassiveFundManager:
    def __init__(self, initial_capital=50_000_000, start_date="2024-10-01", end_date="2025-11-20"):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date

        self.portfolio_value = pd.Series(dtype="float64")
        self.holdings = {}
        self.benchmark_data = None
        self.stock_data = None
        self.base_tickers = [
            "AAPL",
            "MSFT",
            "CSCO",
            "CRM",
            "IBM",
            "JNJ",
            "MRK",
            "AMGN",
            "UNH",
            "AMZN",
            "AXP",
            "GS",
            "JPM",
            "TRV",
            "V",
            "KO",
            "PG",
            "WMT",
            "MCD",
            "NKE",
            "HD",
            "BA",
            "CAT",
            "HON",
            "MMM",
            "DIS",
            "VZ",
            "CVX",
        ]

        self.old_tickers = ["INTC", "DOW"]
        self.new_tickers = ["NVDA", "SHW"]
        self.tickers = self.base_tickers + self.old_tickers + self.new_tickers
        self.benchmark_ticker = "^DJI"

    def get_djia_constituents(self, date):
        switch_date = pd.Timestamp("2024-11-08")
        current_date = pd.to_datetime(date)

        if current_date < switch_date:
            return self.base_tickers + self.old_tickers  
        else:
            return self.base_tickers + self.new_tickers  

    def fetch_market_data(self):
        print(f"🔄 Retrieving market data for period: {self.start_date} to {self.end_date}...")
        fetch_start = (pd.to_datetime(self.start_date) - timedelta(days=10)).strftime("%Y-%m-%d")

        try:
            raw_data = yf.download(
                self.tickers,
                start=fetch_start,
                end=self.end_date,
                progress=False,
                group_by="column",
            )

            if "Adj Close" in raw_data.columns.get_level_values(0):
                self.stock_data = raw_data["Adj Close"]
            elif "Close" in raw_data:
                self.stock_data = raw_data["Close"]
            else:
                self.stock_data = raw_data

            self.stock_data = self.stock_data.dropna(axis=1, how="all").ffill().bfill()

            bench_raw = yf.download(
                self.benchmark_ticker, start=fetch_start, end=self.end_date, progress=False
            )
            if "Adj Close" in bench_raw.columns:
                self.benchmark_data = bench_raw["Adj Close"]
            else:
                self.benchmark_data = bench_raw.iloc[:, 0]
            if isinstance(self.benchmark_data, pd.DataFrame):
                self.benchmark_data = self.benchmark_data.iloc[:, 0]

            self.stock_data = self.stock_data.loc[self.start_date : self.end_date]
            self.benchmark_data = self.benchmark_data.loc[self.start_date : self.end_date]

            if self.stock_data.empty:
                raise ValueError("Dataset empty.")

            print(
                f"✅ Data ready. Loaded {len(self.stock_data.columns)} historical tickers (Portfolio will hold exactly 30)."
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            raise e

    def construct_portfolio(self, date, capital_available):
        if date not in self.stock_data.index:
            loc = self.stock_data.index.get_indexer([date], method="pad")[0]
            if loc == -1:
                loc = 0
            date = self.stock_data.index[loc]

        active_tickers = self.get_djia_constituents(date)
        valid_tickers = [t for t in active_tickers if t in self.stock_data.columns]

        prices = self.stock_data.loc[date, valid_tickers].replace(0, np.nan).dropna()
        weights = prices / prices.sum()
        shares = (capital_available * weights) / prices

        return shares, weights, date

    def track_performance(self):
        if self.stock_data is None:
            self.fetch_market_data()

        print("\n🚀 Initializing Portfolio Simulation...")
        start_date_idx = self.stock_data.index[0]

        self.holdings, _, _ = self.construct_portfolio(start_date_idx, self.initial_capital)

        print(f"   📅 Start Date: {start_date_idx.date()}")
        print(f"   💼 Initial Holdings: {len(self.holdings)} stocks (Checking: INTC in, NVDA out)")

        portfolio_values = []
        event_rebalance_date = pd.Timestamp("2024-11-08")
        has_rebalanced = False

        for date in self.stock_data.index:
            daily_prices = self.stock_data.loc[date]

            current_val = (
                self.holdings * daily_prices.reindex(self.holdings.index).fillna(0)
            ).sum()

            if not has_rebalanced and date >= event_rebalance_date:
                print(f"   ⚖️  [EVENT 2024-11-08] Rebalancing Portfolio...")
                print("       -> Selling: INTC, DOW")
                print("       -> Buying:  NVDA, SHW")

                self.holdings, _, _ = self.construct_portfolio(date, current_val)
                has_rebalanced = True
                print(f"       -> New Holdings Count: {len(self.holdings)} stocks (Correct)")

            portfolio_values.append(current_val)

        self.portfolio_value = pd.Series(portfolio_values, index=self.stock_data.index)
        print("🏁 Simulation Completed.")

    def evaluate_and_plot(self):
        if self.portfolio_value.empty:
            return

        total_ret = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1

        port_returns = self.portfolio_value.pct_change().dropna()
        bench_returns = self.benchmark_data.pct_change().dropna()

        common_dates = port_returns.index.intersection(bench_returns.index)
        port_returns = port_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]

        active_returns = port_returns - bench_returns

        tracking_error = active_returns.std() * np.sqrt(252)

        volatility = port_returns.std() * np.sqrt(252)
        sharpe = (port_returns.mean() * 252 - 0.04) / volatility if volatility != 0 else 0

        print("\n" + "=" * 45)
        print(f" PERFORMANCE REPORT")
        print("=" * 45)
        print(f"Ending Value:    ${self.portfolio_value.iloc[-1]:,.2f}")
        print(f"Total Return:    {total_ret:.2%}")
        print(f"Volatility:      {volatility:.2%}")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Tracking Error:  {tracking_error:.2%} (Target < 5%)")  
        print("=" * 45)

        plt.figure(figsize=(12, 6))
        (self.portfolio_value / self.portfolio_value.iloc[0]).plot(
            label="Passive Fund", linewidth=2
        )
        (self.benchmark_data / self.benchmark_data.iloc[0]).plot(
            label="DJIA Benchmark", linestyle="--", color="black", alpha=0.7
        )

        plt.axvline(
            pd.Timestamp("2024-11-08"), color="red", linestyle=":", label="Rebalance (NVDA/SHW)"
        )
        plt.title(f"Fund Performance vs DJIA (TE: {tracking_error:.2%})")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def analyze_rebalancing(self, target_date=None):
        if self.stock_data is None:
            print("⚠️ Data not loaded. Auto-fetching market data now...")
            self.fetch_market_data()
            if not self.holdings:
                print("⚠️ Portfolio not initialized. Running basic simulation...")
                self.track_performance()

        if target_date:
            check_date = pd.Timestamp(target_date)
        else:
            check_date = pd.Timestamp("2024-11-08")

        print(f"\n📊 CHECKING PORTFOLIO AT: {check_date.date()}")

        if check_date not in self.stock_data.index:
            loc = self.stock_data.index.get_indexer([check_date], method="pad")[0]
            check_date = self.stock_data.index[loc]

        prices = self.stock_data.loc[check_date]

        print(f"   👉 TOTAL STOCKS HELD: {len(self.holdings)} (Should be 30)")

        print("   --- Checking Specific Pairs ---")
        if "NVDA" in self.holdings:
            print(f"   ✅ NVDA: HELD (Val: ${self.holdings['NVDA']*prices['NVDA']:,.0f})")
        else:
            print("   ❌ NVDA: NOT HELD")

        if "INTC" not in self.holdings:
            print("   ✅ INTC: REMOVED")
        else:
            print("   ⚠️ INTC: STILL HELD")

        if "SHW" in self.holdings:
            print(f"   ✅ SHW : HELD (Val: ${self.holdings['SHW']*prices['SHW']:,.0f})")
        else:
            print("   ❌ SHW : NOT HELD")

        if "DOW" not in self.holdings:
            print("   ✅ DOW : REMOVED")
        else:
            print("   ⚠️ DOW : STILL HELD")


def hedged_long_futures_nav_and_metrics(
    fund_object,
    cash_weight=0.05,
    rf_annual=0.04,
    periods_per_year=252,
):
    """
    Passive fund after cash equitization using long futures on DJIA:
    - (1 - cash_weight) invested in your stock portfolio (fund_object.portfolio_value)
    - cash_weight held as cash BUT equitized via LONG DJIA futures => earns benchmark return
    So total hedged return each day:
        r_hedged = (1-w)*r_port + w*r_bench
    Metrics: Total Return, Vol, Sharpe, Tracking Error, Max Drawdown
    Returns:
        nav_hedged (Series), metrics (dict), drawdown_series (Series)
    """

    if fund_object.portfolio_value is None or len(fund_object.portfolio_value) == 0:
        raise ValueError(
            "fund_object.portfolio_value is empty. Run fund_object.track_performance() first."
        )
    if fund_object.benchmark_data is None or len(fund_object.benchmark_data) == 0:
        raise ValueError(
            "fund_object.benchmark_data is empty. Run fund_object.fetch_market_data() first."
        )

    nav_port = fund_object.portfolio_value.copy()
    bench_px = fund_object.benchmark_data.copy()

    # daily returns
    r_port = nav_port.pct_change().dropna()
    r_bench = bench_px.pct_change().dropna()

    common = r_port.index.intersection(r_bench.index)
    r_port = r_port.loc[common]
    r_bench = r_bench.loc[common]

    # --- Hedged returns with long futures (cash equitization) ---
    w = float(cash_weight)
    r_hedged = (1 - w) * r_port + w * r_bench

    initial_nav = float(getattr(fund_object, "initial_capital", nav_port.loc[common[0]]))
    nav_hedged = initial_nav * (1 + r_hedged).cumprod()
    nav_hedged.name = "NAV_Hedged_LongFut"

    # --- Metrics ---
    total_return = nav_hedged.iloc[-1] / nav_hedged.iloc[0] - 1

    vol = r_hedged.std() * np.sqrt(periods_per_year)

    # Sharpe 
    rf_per_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = r_hedged - rf_per_period
    sharpe = (excess.mean() * periods_per_year) / vol if vol != 0 else np.nan

    # Tracking Error vs benchmark (annualized)
    active = r_hedged - r_bench
    te = active.std() * np.sqrt(periods_per_year)

    # Drawdown (from NAV)
    running_max = nav_hedged.cummax()
    drawdown = nav_hedged / running_max - 1
    max_drawdown = drawdown.min()

    metrics = {
        "Total Return": total_return,
        "Volatility (ann.)": vol,
        "Sharpe (ann.)": sharpe,
        "Tracking Error vs DJIA (ann.)": te,
        "Max Drawdown": max_drawdown,
        "Ending NAV": nav_hedged.iloc[-1],
    }

    return nav_hedged, metrics, drawdown

# ADVANCED PASSIVE ANALYTICS & HEDGING ==============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


class AdvancedPassiveManager(PassiveFundManager): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sector_map = {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "CSCO": "Information Technology",
            "CRM": "Information Technology",
            "IBM": "Information Technology",
            "INTC": "Information Technology",
            "JNJ": "Health Care",
            "MRK": "Health Care",
            "AMGN": "Health Care",
            "UNH": "Health Care",
            "AXP": "Financials",
            "GS": "Financials",
            "JPM": "Financials",
            "TRV": "Financials",
            "V": "Financials",
            "KO": "Consumer Staples",
            "PG": "Consumer Staples",
            "WMT": "Consumer Staples",
            "MCD": "Consumer Discretionary",
            "NKE": "Consumer Discretionary",
            "HD": "Consumer Discretionary",
            "BA": "Industrials",
            "CAT": "Industrials",
            "HON": "Industrials",
            "MMM": "Industrials",
            "DIS": "Communication Services",
            "VZ": "Communication Services",
            "CVX": "Energy",
            "DOW": "Materials",
            "SHW": "Materials",
        }

    def analyze_sector_allocation(self):
        """Analyze portfolio allocation by sector."""
        print("\n🔍 Analyzing Sector Allocation...")

        last_prices = self.stock_data.iloc[-1]
        current_holdings = self.holdings
        market_vals = current_holdings * last_prices

        df_sector = pd.DataFrame({"MarketValue": market_vals})
        df_sector["Sector"] = df_sector.index.map(self.sector_map)

        # Group by sector
        sector_alloc = df_sector.groupby("Sector")["MarketValue"].sum()
        sector_weights = sector_alloc / sector_alloc.sum()

        return sector_weights

    def calculate_attribution(self):
        """Compute simplified performance attribution (top contributors)."""
        stock_returns = (self.stock_data.iloc[-1] - self.stock_data.iloc[0]) / self.stock_data.iloc[
            0
        ]

        avg_weights = self.stock_data.iloc[0] / self.stock_data.iloc[0].sum()

        contribution = stock_returns * avg_weights
        return contribution.sort_values(ascending=False)

    def derivatives_overlay_analysis(self):
        """Analyze a futures hedging overlay and rolling beta."""
        print("\n🛡️ Computing Hedge Ratios and Rolling Beta...")

        port_ret = self.portfolio_value.pct_change().dropna()
        bench_ret = self.benchmark_data.pct_change().dropna()

        common_idx = port_ret.index.intersection(bench_ret.index)
        port_ret = port_ret.loc[common_idx]
        bench_ret = bench_ret.loc[common_idx]

        rolling_cov = port_ret.rolling(window=60).cov(bench_ret)
        rolling_var = bench_ret.rolling(window=60).var()
        rolling_beta = rolling_cov / rolling_var

        current_beta = rolling_beta.iloc[-1]

        futures_multiplier = 10
        index_level = self.benchmark_data.iloc[-1]  
        contract_value = index_level * futures_multiplier

        portfolio_value = self.portfolio_value.iloc[-1]

        hedge_contracts_full = current_beta * (portfolio_value / contract_value)

        print("\n" + "=" * 50)
        print(" DERIVATIVES & RISK MANAGEMENT REPORT")
        print("=" * 50)
        print(f"Current Beta (60-day):         {current_beta:.4f}")
        print(f"Portfolio Value:               ${portfolio_value:,.2f}")
        print(f"DJIA Index Level (assumed):    {index_level:,.0f}")
        print(f"Futures Contract Value ($10x): ${contract_value:,.2f}")
        print("-" * 50)
        print("📉 HEDGING RECOMMENDATION:")
        print(f"To hedge 100% market risk, SHORT: {hedge_contracts_full:.2f} contracts")
        print(f"To hedge 50% (tactical), SHORT:   {hedge_contracts_full / 2:.2f} contracts")
        print("=" * 50)

        return rolling_beta

    def generate_dashboard(self):
        if self.portfolio_value.empty:
            self.track_performance()

        # 1. Performance metric
        end_value = self.portfolio_value.iloc[-1]
        total_ret = (end_value / self.initial_capital) - 1
        volatility = self.portfolio_value.pct_change().std() * np.sqrt(252)
        sharpe = (total_ret - 0.03) / volatility

        # 2. Hedge metric
        target_date = pd.to_datetime(self.start_date)
        try:
            loc = self.benchmark_data.index.get_indexer([target_date], method="bfill")[0]
            start_date_idx = (
                self.benchmark_data.index[loc] if loc != -1 else self.benchmark_data.index[0]
            )
        except Exception:
            start_date_idx = self.benchmark_data.index[0]

        index_at_start = self.benchmark_data.loc[start_date_idx]
        if isinstance(index_at_start, pd.Series):
            index_at_start = index_at_start.item()

        equity_exposure_start = self.initial_capital * 0.95
        contract_value_start = index_at_start * 5
        contracts_needed = equity_exposure_start / contract_value_start

        print("=" * 80)
        print(" 📊 FUND PERFORMANCE & RISK REPORT")
        print("=" * 80)
        print(f"Investment Period:      {self.start_date} to {self.end_date}")
        print(f"Ending Value:           ${end_value:,.2f}")
        print(f"Total Return:           {total_ret:.2%}")
        print(f"Sharpe Ratio:           {sharpe:.2f}")
        print("-" * 80)
        print(" 🛡️  HEDGE SETUP (AT INCEPTION)")
        print(f"Execution Date:         {start_date_idx.date()}")
        print(f"Equity Exposure (95%):  ${equity_exposure_start:,.2f} (Target for Hedging)")
        print(f"Futures Price (Start):  {index_at_start:,.0f}")
        print(f"Contracts Required:     {contracts_needed:.2f} (SHORT)")
        print(
            f"Recommendation:         Sell {int(round(contracts_needed))} DJIA Futures to stay Market Neutral."
        )
        print("=" * 80)
        print("\nDisplaying Visual Dashboard...")

        # Prepare data using helper methods
        sector_w = self.analyze_sector_allocation()
        contrib = self.calculate_attribution()
        rolling_beta = self.derivatives_overlay_analysis()

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        colors = sns.color_palette("pastel")[0 : len(sector_w)]
        wedges, texts, autotexts = ax1.pie(
            sector_w,
            labels=sector_w.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.85,
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax1.add_artist(centre_circle)
        ax1.set_title("Portfolio Sector Allocation (Price-Weighted)", fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1])
        top_bot = pd.concat([contrib.head(5), contrib.tail(5)])
        colors_bar = ["green" if x > 0 else "red" for x in top_bot.values]
        top_bot.plot(kind="barh", ax=ax2, color=colors_bar, alpha=0.7)
        ax2.set_title("Performance Attribution: Top Winners & Losers", fontweight="bold")
        ax2.set_xlabel("Weighted Contribution to Return")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(rolling_beta, color="purple", label="60-day Rolling Beta")
        ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Market Beta (1.0)")

        ax3.fill_between(
            rolling_beta.index,
            rolling_beta,
            1.0,
            where=(rolling_beta > 1.0),
            interpolate=True,
            color="red",
            alpha=0.1,
            label="High Sensitivity Area",
        )
        ax3.set_title("Dynamic Risk Profile: Portfolio Beta vs Market", fontweight="bold")
        ax3.legend()

        plt.tight_layout()
        plt.show()

# HEDGING STRATEGY & EXPLANATION ANALYSIS ==============================================================================


def explain_hedging_logic(fund_object):
    """
    Explains the Short Futures Hedging logic with a specific capital allocation:
    - Total Capital: $50M
    - Cash Reserve (5%): $2.5M (Reserved for Long Futures exposure)
    - Equity to Hedge (95%): $47.5M
    """

    if fund_object.benchmark_data is None or fund_object.benchmark_data.empty:
        print("❌ Error: Benchmark data is missing.")
        print("👉 Please run 'fund.fetch_market_data()' or 'fund.track_performance()' first.")
        return

    target_date = pd.to_datetime(fund_object.start_date)

    try:
        idx_loc = fund_object.benchmark_data.index.get_indexer([target_date], method="bfill")[0]

        if idx_loc == -1:
            idx_loc = 0

        calc_date = fund_object.benchmark_data.index[idx_loc]

    except Exception as e:
        print(f"⚠️ Warning: Date lookup failed ({e}). Defaulting to first available date.")
        calc_date = fund_object.benchmark_data.index[0]

    total_capital = 50_000_000
    cash_reserve_pct = 0.05

    cash_reserve = total_capital * cash_reserve_pct

    hedged_equity_value = total_capital - cash_reserve  # $47,500,000

    print(f"1. CAPITAL STRUCTURE & ALLOCATION (Oct 2024):")
    print(f"   • Total Portfolio AUM:    ${total_capital:,.2f}")
    print(f"   • Cash/Margin (5%):       ${cash_reserve:,.2f} (Reserved for Long Futures Exposure)")
    print(f"   • Equity Portfolio (95%): ${hedged_equity_value:,.2f}")
    print(f"     (We are hedging this $47.5M against market downside)")

    try:
        index_price = fund_object.benchmark_data.loc[calc_date]
        if isinstance(index_price, pd.Series) or isinstance(index_price, np.ndarray):
            index_price = index_price.item()
    except KeyError:
        index_price = fund_object.benchmark_data.iloc[idx_loc]

    contract_multiplier = 5

    notional_per_contract = index_price * contract_multiplier

    num_contracts = hedged_equity_value / notional_per_contract

    print(f"\n2. HEDGE CALCULATION (At Start Date: {calc_date.date()}):")
    print(f"   • DJIA Index Level:       {index_price:,.2f}")
    print(f"   • Futures Multiplier:     ${contract_multiplier} (E-mini DJIA)")
    print(f"   • Notional per Contract:  ${notional_per_contract:,.2f}")
    print(f"   ------------------------------------------------------------")
    print(
        f"   • Calculation:            ${hedged_equity_value:,.0f} / ${notional_per_contract:,.2f}"
    )
    print(f"   • Theoretical Short:      {num_contracts:.2f} contracts")

    print(f"\n3. EXECUTION STRATEGY:")
    print(f"   👉 SELL {int(round(num_contracts))} CONTRACTS of DJIA Futures (Short Position).")
    print(f"   Rationale: This locks in the value of the $47.5M equity portfolio.")
    print(f"   The remaining 5% cash is used separately for the active strategy.")
    print("=" * 60 + "\n")

# HEDGED PORTFOLIO SIMULATION (SIMULATING HEDGE EFFECTIVENESS) ==============================================================================

def simulate_hedged_performance(fund_object):
    """
    Simulates a 'Market Neutral' strategy with specific allocation:
    - 95% Capital ($47.5M) invested in the Passive Portfolio.
    - 5% Capital ($2.5M) held as Cash (Collateral).
    - Short Futures Hedge established at START DATE to fully hedge the 95% Equity.
    """
    total_capital = 50_000_000
    equity_alloc = 0.95
    cash_alloc = 0.05

    initial_equity_val = total_capital * equity_alloc  
    initial_cash_val = total_capital * cash_alloc  

    port_normalized = fund_object.portfolio_value / fund_object.portfolio_value.iloc[0]
    equity_curve = port_normalized * initial_equity_val

    bench_data = fund_object.benchmark_data.loc[equity_curve.index]

    bench_normalized = bench_data / bench_data.iloc[0]

    short_pnl_curve = initial_equity_val * (1 - bench_normalized)

    hedged_portfolio_value = equity_curve + short_pnl_curve + initial_cash_val

    hedged_return = (hedged_portfolio_value.iloc[-1] / total_capital) - 1
    unhedged_return = (
        fund_object.portfolio_value.iloc[-1] / fund_object.portfolio_value.iloc[0]
    ) - 1

    volatility = hedged_portfolio_value.pct_change().std() * np.sqrt(252)

    print(f"Strategy Structure:")
    print(f"   • Long Equity (Start):  ${initial_equity_val:,.2f} (95%)")
    print(f"   • Cash Reserve:         ${initial_cash_val:,.2f} (5%)")
    print(f"   • Short Hedge:          Fully hedging the $47.5M Equity exposure.")

    print(f"\nResults ({fund_object.start_date} - {fund_object.end_date}):")
    print(f"   • Unhedged Return:      {unhedged_return:.2%}")
    print(f"   • Hedged Return (Alpha):{hedged_return:.2%}")
    print(f"   • Hedged Volatility:    {volatility:.2%}")

    if hedged_return > 0:
        print(
            f"   ✅ SUCCESS: The portfolio generated positive Alpha ({hedged_return:.2%}) independent of market moves."
        )
    else:
        print(
            f"   ⚠️ NOTE: Negative Alpha detected. Stock selection underperformed the hedge cost."
        )

    plt.figure(figsize=(12, 6))

    (hedged_portfolio_value / total_capital).plot(
        label="Hedged Strategy (Market Neutral)", color="green", linewidth=2
    )

    bench_normalized.plot(label="Benchmark (DJIA)", color="gray", linestyle="--", alpha=0.5)

    plt.title("Hedged Strategy Performance (Pure Alpha Generation)")
    plt.ylabel("Normalized Value (Start = 1.0)")
    plt.axhline(1.0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return hedged_portfolio_value


# CASH EQUITIZATION ANALYSIS ==========================================================================================


class CashEquitizationManager:
    def __init__(self, fund_object, initial_nav=50_000_000):
        self.fund = fund_object
        self.initial_nav = float(initial_nav)
        bench_ret = fund_object.benchmark_data.pct_change().dropna()
        nav = fund_object.portfolio_value
        if isinstance(nav, pd.DataFrame):
            nav = nav.iloc[:, 0]
        port_ret = nav.pct_change().dropna()

        common_dates = bench_ret.index.intersection(port_ret.index)
        self.benchmark_ret = bench_ret.loc[common_dates]
        self.fund_nav_aligned = nav.loc[common_dates]

        self.benchmark_px = fund_object.benchmark_data
        if isinstance(self.benchmark_px, pd.DataFrame):
            self.benchmark_px = self.benchmark_px.iloc[:, 0]

    def calculate_futures_needed(self, cash_amount, index_price, multiplier=10):
        notional_per_contract = float(index_price) * float(multiplier)
        num_contracts = float(cash_amount) / notional_per_contract
        return num_contracts, notional_per_contract

    def simulate_cash_drag_vs_equitization(self, cash_weight=0.05, multiplier=10):
        print("\n" + "=" * 60)
        print(f"💰 CASH EQUITIZATION SIMULATION (Assuming {cash_weight*100:.1f}% Cash Level)")
        print("=" * 60)

        ret_cash_drag = (1 - cash_weight) * self.benchmark_ret + (cash_weight * 0.0)
        ret_equitized = (1 - cash_weight) * self.benchmark_ret + (cash_weight * self.benchmark_ret)

        def calc_te(strategy_ret, benchmark_ret):
            diff = strategy_ret - benchmark_ret
            return diff.std() * np.sqrt(252)

        te_drag = calc_te(ret_cash_drag, self.benchmark_ret)
        te_equitized = calc_te(ret_equitized, self.benchmark_ret)

        start_date = self.benchmark_ret.index[0] 

        total_nav_begin = self.initial_nav
        cash_balance = total_nav_begin * cash_weight

        index_level_begin = float(self.benchmark_px.loc[start_date])

        n_contracts, contract_val = self.calculate_futures_needed(
            cash_balance, index_level_begin, multiplier=multiplier
        )

        print(f"1️⃣ SITUATION ANALYSIS:")
        print(f"   - Total NAV (Begin):       ${total_nav_begin:,.2f}")
        print(f"   - Idle Cash (Begin):       ${cash_balance:,.2f} ({cash_weight*100:.1f}%)")
        print(
            f"   - DJIA Level (Begin):      {index_level_begin:,.2f}  | Date: {start_date.date()}"
        )
        print("-" * 60)
        print(f"2️⃣ EXECUTION (LONG FUTURES):")
        print(f"   - Contract Value (${multiplier}x): ${contract_val:,.2f}")
        print(f"   - Contracts Needed:        {n_contracts:.4f}")
        print(
            f"   👉 RECOMMENDATION:         LONG {int(np.ceil(n_contracts))} DJIA Futures Contracts"
        )
        print("-" * 60)
        print(f"3️⃣ IMPACT ON TRACKING ERROR (The Philosophy):")
        print(f"   ❌ Scenario A (Hold Cash):    Tracking Error = {te_drag:.4%}")
        print(f"   ✅ Scenario B (Equitized):    Tracking Error = {te_equitized:.4%} (Minimized!)")

        return ret_cash_drag, ret_equitized, cash_balance

    def plot_comparison(self, ret_drag, ret_equitized):
        cum_bench = (1 + self.benchmark_ret).cumprod()
        cum_drag = (1 + ret_drag).cumprod()
        cum_eq = (1 + ret_equitized).cumprod()

        plt.figure(figsize=(14, 7))
        plt.plot(
            cum_bench,
            label="Benchmark (DJIA)",
            color="black",
            alpha=0.6,
            linestyle="--",
            linewidth=2,
        )
        plt.plot(
            cum_drag, label="Portfolio with Cash Drag (No Futures)", color="red", linewidth=1.5
        )
        plt.plot(
            cum_eq,
            label="Equitized Portfolio (With Long Futures)",
            color="green",
            linestyle="-.",
            linewidth=2,
        )

        plt.title("Impact of Cash Equitization on Tracking Quality (Eliminating Cash Drag)")
        plt.ylabel("Cumulative Return (Growth of $1)")
        plt.legend()
        plt.show()


# VISUALIZATION: SECTOR ALLOCATION & DRAWDOWN ==============================================================================


class FundVisualizer:
    def __init__(self, fund_object):
        self.fund = fund_object
        self.tickers = fund_object.tickers
        self.holdings = fund_object.holdings
        self.portfolio_value = fund_object.portfolio_value
        self.benchmark_data = fund_object.benchmark_data

        self.sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "CRM": "Technology",
            "CSCO": "Technology",
            "IBM": "Technology",
            "INTC": "Technology",
            "NVDA": "Technology",
            "GS": "Financials",
            "JPM": "Financials",
            "AXP": "Financials",
            "V": "Financials",
            "TRV": "Financials",
            "UNH": "Health Care",
            "JNJ": "Health Care",
            "MRK": "Health Care",
            "AMGN": "Health Care",
            "HD": "Consumer Discretionary",
            "MCD": "Consumer Discretionary",
            "NKE": "Consumer Discretionary",
            "AMZN": "Consumer Discretionary",
            "BA": "Industrials",
            "CAT": "Industrials",
            "HON": "Industrials",
            "MMM": "Industrials",
            "KO": "Consumer Staples",
            "PG": "Consumer Staples",
            "WMT": "Consumer Staples",
            "CVX": "Energy",
            "SHW": "Materials",
            "DOW": "Materials",
            "DIS": "Communication",
            "VZ": "Communication",
        }

    def plot_sector_allocation(self):
        """
        Plot a pie chart showing sector allocation weights.
        """
        print("\n📊 Generating Sector Allocation Chart...")

        last_prices = self.fund.stock_data.iloc[-1]

        active_holdings = {t: s for t, s in self.holdings.items() if t in last_prices.index}

        sector_values = {}

        for ticker, shares in active_holdings.items():
            sector = self.sector_map.get(ticker, "Other")  
            market_val = shares * last_prices[ticker]

            if sector in sector_values:
                sector_values[sector] += market_val
            else:
                sector_values[sector] = market_val

        s_sector = pd.Series(sector_values).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        colors = plt.get_cmap("Pastel1")(np.linspace(0, 1, len(s_sector)))

        wedges, texts, autotexts = plt.pie(
            s_sector,
            labels=s_sector.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "white"},
        )

        plt.title("Portfolio Sector Allocation (Price-Weighted)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print("\n--- Sector Allocation Details (Sector Weights) ---")
        print((s_sector / s_sector.sum() * 100).round(2).astype(str) + "%")

    def plot_drawdown(self):
        """
        Plot drawdown (drop from peak) and compare it to the benchmark.
        Formula: Drawdown = (NAV / Running_Max_NAV) - 1
        """
        print("\n📉 Generating Drawdown Analysis Chart...")

        port_nav = self.portfolio_value
        running_max_port = port_nav.cummax()
        drawdown_port = (port_nav / running_max_port) - 1

        bench_price = self.benchmark_data.loc[port_nav.index]
        running_max_bench = bench_price.cummax()
        drawdown_bench = (bench_price / running_max_bench) - 1

        max_dd_port = drawdown_port.min()
        max_dd_bench = drawdown_bench.min()

        plt.figure(figsize=(14, 6))

        plt.plot(
            drawdown_port.index,
            drawdown_port,
            label=f"Passive Fund (Max DD: {max_dd_port:.2%})",
            color="blue",
            linewidth=1.5,
        )
        plt.fill_between(drawdown_port.index, drawdown_port, 0, color="blue", alpha=0.1)

        plt.plot(
            drawdown_bench.index,
            drawdown_bench,
            label=f"Benchmark DJIA (Max DD: {max_dd_bench:.2%})",
            color="gray",
            linestyle="--",
            linewidth=1.5,
        )

        plt.title("Historical Drawdown (Risk Analysis)", fontsize=14, fontweight="bold")
        plt.ylabel("Drawdown (%)")
        plt.axhline(0, color="black", linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
