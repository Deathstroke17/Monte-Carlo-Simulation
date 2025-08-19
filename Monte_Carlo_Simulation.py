import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -- 1. DATA ACQUISITION & PREPARATION --

# Define the assets (stock tickers) and the time period for historical data.
# You can change these to any stocks you are interested in.
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

# Download adjusted closing prices from Yahoo Finance
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Close']

# Calculate daily logarithmic returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna() # Drop the first row with NaN values

# -- 2. MONTE CARLO SIMULATION SETUP --

# Calculate mean returns and covariance matrix for the assets
mean_returns = log_returns.mean() * 252 # Annualize
cov_matrix = log_returns.cov() * 252   # Annualize

# Set simulation parameters
num_portfolios = 20000
risk_free_rate = 0.02 # Example risk-free rate (e.g., 10-year Treasury yield)

# Create arrays to store simulation results
portfolio_returns = []
portfolio_volatility = []
portfolio_sharpe_ratios = []
portfolio_weights = []

# -- 3. RUN THE SIMULATION --

print("Running Monte Carlo Simulation...")

for _ in range(num_portfolios):
    # 1. Generate random weights
    weights = np.random.random(len(tickers))
    # 2. Normalize weights so they sum to 1
    weights /= np.sum(weights)
    portfolio_weights.append(weights)
    
    # 3. Calculate portfolio's expected annual return
    port_return = np.sum(mean_returns * weights)
    portfolio_returns.append(port_return)
    
    # 4. Calculate portfolio's expected annual volatility (risk)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_volatility.append(port_volatility)
    
    # 5. Calculate the Sharpe Ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    portfolio_sharpe_ratios.append(sharpe_ratio)

# Combine the results into a single dictionary
portfolio_data = {
    'Return': portfolio_returns,
    'Volatility': portfolio_volatility,
    'Sharpe Ratio': portfolio_sharpe_ratios
}
for i, ticker in enumerate(tickers):
    portfolio_data[f'Weight_{ticker}'] = [w[i] for w in portfolio_weights]

# Create a DataFrame from the results
results_df = pd.DataFrame(portfolio_data)

print("Simulation Complete!")

# -- 4. IDENTIFY OPTIMAL PORTFOLIOS --

# Find the portfolio with the highest Sharpe Ratio
max_sharpe_portfolio = results_df.loc[results_df['Sharpe Ratio'].idxmax()]

# Find the portfolio with the minimum volatility (lowest risk)
min_volatility_portfolio = results_df.loc[results_df['Volatility'].idxmin()]

print("\n" + "="*80)
print("Optimal Portfolio Analysis:")
print("="*80)

print("\n Maximum Sharpe Ratio Portfolio (Best Risk-Adjusted Return):")
print(f"   - Annualized Return: {max_sharpe_portfolio['Return']:.2%}")
print(f"   - Annualized Volatility: {max_sharpe_portfolio['Volatility']:.2%}")
print(f"   - Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
print("\n   Optimal Asset Allocation:")
for ticker in tickers:
    print(f"     - {ticker}: {max_sharpe_portfolio[f'Weight_{ticker}']:.2%}")

print("\n" + "-"*80)

print("\n Minimum Volatility Portfolio (Lowest Risk):")
print(f"   - Annualized Return: {min_volatility_portfolio['Return']:.2%}")
print(f"   - Annualized Volatility: {min_volatility_portfolio['Volatility']:.2%}")
print(f"   - Sharpe Ratio: {min_volatility_portfolio['Sharpe Ratio']:.2f}")
print("\n   Optimal Asset Allocation:")
for ticker in tickers:
    print(f"     - {ticker}: {min_volatility_portfolio[f'Weight_{ticker}']:.2%}")
print("\n" + "="*80)


# -- 5. VISUALIZATION --

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(12, 7))
# Create a scatter plot of all simulated portfolios
# Color-map the points by their Sharpe Ratio for better visual analysis
sc = plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o', s=10, alpha=0.7)

plt.title('Monte Carlo Simulation - Efficient Frontier', fontsize=18)
plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
plt.ylabel('Annualized Return', fontsize=12)

# Highlight the max Sharpe ratio portfolio
plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], marker='*', color='r', s=500, label='Max Sharpe Ratio', edgecolors='black')
# Highlight the min volatility portfolio
plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return'], marker='X', color='orange', s=300, label='Min Volatility', edgecolors='black')

plt.legend(labelspacing=0.8)
cbar = plt.colorbar(sc)
cbar.set_label('Sharpe Ratio')
plt.show()