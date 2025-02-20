import os
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from datetime import timedelta
import matplotlib.ticker as mticker
import plotly.graph_objects as go  # <-- Plotly for visualization

##############################################################################
# 1. PARAMETER SPACE DEFINITION
##############################################################################
space = [
    Integer(75, 85, name='rsi_overbought'),
    Integer(10, 25, name='rsi_oversold'),
    Real(0.03, 0.12, name='trailing_stop_pct'),
    Real(0.02, 0.05, name='stop_loss_pct'),
    Real(0.1, 0.25, name='take_profit_pct')
]

##############################################################################
# 2. INDICATOR CALCULATION FUNCTIONS
##############################################################################
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

##############################################################################
# 3. DATA LOADING AND CLEANING
##############################################################################
def load_data(filepath, ticker="MSFT", period="2y", interval="1h"):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Downloading stock data for {ticker}...")
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Error: No data downloaded for {ticker}. Check if market is open.")
        df.reset_index(inplace=True)
        df.to_csv(filepath, index=False)
        print(f"Stock data saved to {filepath}.")
    else:
        print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['Date'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(how='any', inplace=True)

    # Compute indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])

    return df

##############################################################################
# 4. SIGNAL GENERATION (WITH TRADE DURATION & PRINTS)
##############################################################################
def generate_signals(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    signals = pd.DataFrame(index=data.index)
    signals['Position'] = 0
    signals['Trailing_Stop'] = np.nan

    trade_durations = []
    trade_start_time = None
    buy_signals, sell_signals = [], []

    for i in range(1, len(data)):
        close_price = data['Close'].iloc[i]

        # Debug logging every 100th row
        if i % 100 == 0:
            print(f"Row {i}: Close={close_price}, RSI={data['RSI'].iloc[i]}, BB_Lower={data['BB_lower'].iloc[i]}")

        # Long Entry
        if (data['RSI'].iloc[i] < rsi_oversold) and (close_price < data['BB_lower'].iloc[i]):
            if trade_start_time is None:  # Only record if no trade open
                trade_start_time = data.index[i]
            signals.loc[data.index[i], 'Position'] = 1

            # Print buy event in console
            print(f"BUY at {data.index[i]}: RSI={data['RSI'].iloc[i]}, Close={close_price}")
            buy_signals.append(data.index[i])

        # Short Entry (i.e., closing the long trade)
        elif (data['RSI'].iloc[i] > rsi_overbought) and (close_price > data['BB_upper'].iloc[i]):
            if trade_start_time is not None:
                # Close the trade; record duration
                trade_duration = data.index[i] - trade_start_time
                trade_durations.append(trade_duration)
                trade_start_time = None
            signals.loc[data.index[i], 'Position'] = -1

            # Print sell event in console
            print(f"SELL at {data.index[i]}: RSI={data['RSI'].iloc[i]}, Close={close_price}")
            sell_signals.append(data.index[i])

    # Compute average trade duration
    avg_trade_duration = sum(trade_durations, timedelta(0)) / len(trade_durations) if trade_durations else timedelta(0)

    return signals, buy_signals, sell_signals, avg_trade_duration

##############################################################################
# 5. PERFORMANCE METRICS & WALK-FORWARD ANALYSIS
##############################################################################
def walk_forward_analysis(data, rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    signals, buy_signals, sell_signals, avg_trade_duration = generate_signals(
        data, rsi_overbought, rsi_oversold, trailing_stop_pct,
        stop_loss_pct, take_profit_pct
    )
    returns = data['Close'].pct_change()
    signals['Strategy_Returns'] = returns * signals['Position'].shift()
    cumulative_returns = (1 + signals['Strategy_Returns']).cumprod() - 1

    if cumulative_returns.empty:
        # Return defaults if no trades
        from datetime import timedelta
        return -999, 0, 0, 0, timedelta(0), signals, buy_signals, sell_signals

    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    std_dev = signals['Strategy_Returns'].std()
    if std_dev != 0:
        sharpe_ratio = (signals['Strategy_Returns'].mean() / std_dev) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    adjusted_sharpe = sharpe_ratio - max_drawdown
    active_trades = (signals['Strategy_Returns'] != 0).sum()
    if active_trades > 0:
        win_rate = (signals['Strategy_Returns'] > 0).sum() / active_trades
    else:
        win_rate = 0

    return -adjusted_sharpe, cumulative_returns.iloc[-1], max_drawdown, win_rate, avg_trade_duration, signals, buy_signals, sell_signals

##############################################################################
# 6. BAYESIAN OPTIMIZATION
##############################################################################
global_data = None
from skopt.utils import use_named_args

@use_named_args(space)
def objective(rsi_overbought, rsi_oversold, trailing_stop_pct, stop_loss_pct, take_profit_pct):
    global global_data

    if global_data is None:
        global_data = load_data("MSFT_1hour_data.csv")

    adjusted_sharpe, _, _, _, _, _, _, _ = walk_forward_analysis(
        global_data, rsi_overbought, rsi_oversold, trailing_stop_pct,
        stop_loss_pct, take_profit_pct
    )
    return adjusted_sharpe

print("Starting Bayesian optimization...")
res = gp_minimize(objective, space, n_calls=40, random_state=0)
print("Bayesian optimization completed.")

##############################################################################
# 7. RESULTS & VISUALIZATIONS
##############################################################################
best_params = res.x
data = load_data("MSFT_1hour_data.csv")
neg_sharpe, cum_return, max_dd, win_rate, avg_dur, signals, buy_points, sell_points = walk_forward_analysis(
    data, *best_params
)

print("\n=== Optimized Strategy Results ===")
print("Best parameters:", best_params)
print("Sharpe Ratio:", -res.fun)
print("Cumulative Return:", cum_return)
print("Max Drawdown:", max_dd)
print("Win Rate:", win_rate)
print("Avg Trade Duration:", avg_dur)

# (1) PLOTLY Chart for Closing Price & Signals ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index, y=data['Close'],
    mode='lines', name='Close Price',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=buy_points, y=data.loc[buy_points]['Close'],
    mode='markers', name='Buy Signal',
    marker=dict(color='green', symbol='triangle-up', size=10)
))

fig.add_trace(go.Scatter(
    x=sell_points, y=data.loc[sell_points]['Close'],
    mode='markers', name='Sell Signal',
    marker=dict(color='red', symbol='triangle-down', size=10)
))

fig.update_layout(
    title="Combined Walk-Forward Analysis with Buy/Sell Signals",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)
fig.show()

# (2) Skopt Convergence Plot
plt.figure(figsize=(7,5))
plot_convergence(res)
plt.title("Convergence Plot")
plt.xlabel("Number of Calls")
plt.ylabel("Objective Value (Neg. Sharpe Ratio)")
plt.show()

# (3) Partial Dependence Plot - reduce label overlap
plt.figure(figsize=(14,10))        # Increase figure size
plot_objective(res, n_points=150)  # More points for a smoother, more spaced distribution
plt.xticks(fontsize=10, rotation=45)  # Rotate & reduce font to avoid overlap
plt.yticks(fontsize=10, rotation=45)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25) # Extra margins
plt.suptitle("Partial Dependence Plot (Optimized)", fontsize=14)
plt.show()

# (4) Parameter Evaluations Plot
plt.figure(figsize=(14,10))
plot_evaluations(res)
plt.suptitle("Parameter Evaluations", fontsize=14)
plt.show()
