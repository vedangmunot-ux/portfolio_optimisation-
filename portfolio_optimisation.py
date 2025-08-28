import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os

# Settings
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
START_DATE, END_DATE = "2019-01-01", "2024-01-01"
RISK_FREE = 0.06

os.makedirs("outputs", exist_ok=True)

# Download data
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]

# Print debug info
print("Data shape:", data.shape)
print(data.head())

# Calculate returns
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(TICKERS)

# Portfolio performance function
def portfolio_performance(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    returns_port = np.dot(weights, mean_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (returns_port - RISK_FREE) / volatility
    return returns_port, volatility, sharpe_ratio

# Objective function: negative Sharpe
def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

# Portfolio optimization
def optimize_portfolio(weights_func):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(weights_func,
                      num_assets * [1. / num_assets,],
                      args=(mean_returns, cov_matrix),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result

# Run optimization
opt_sharpe = optimize_portfolio(neg_sharpe)
weights = opt_sharpe.x
returns_opt, volatility_opt, sharpe_opt = portfolio_performance(weights, mean_returns, cov_matrix)

print("\nOptimized Weights (Max Sharpe):")
for t, w in zip(TICKERS, weights):
    print(f"{t}: {w:.4f}")

print(f"\nExpected Annual Return: {returns_opt*100:.2f}%")
print(f"Annual Volatility: {volatility_opt*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_opt:.2f}")

# --- Efficient Frontier Simulation ---
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    r, v, s = portfolio_performance(w, mean_returns, cov_matrix)
    results[0,i] = r
    results[1,i] = v
    results[2,i] = s

# Plot and save
plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(volatility_opt, returns_opt, c='red', marker='*', s=200, label='Max Sharpe')
plt.xlabel('Annual Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.savefig("outputs/efficient_frontier.png")  # Save figure
plt.close()  # Close plot so terminal is not blocked

print("\nEfficient Frontier plot saved to 'outputs/efficient_frontier.png'")

