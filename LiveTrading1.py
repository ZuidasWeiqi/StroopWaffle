import datetime
import logging
import time

# Allows asyncio re-entrancy in a single thread
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd

from ib_insync import IB, Crypto, MarketOrder, util

# ------------------------------------------------
# Logging Setup (prints to console)
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ------------------------------------------------
# IBKR Connection Settings
# ------------------------------------------------
HOST = "127.0.0.1"
PORT = 4001          # e.g. 7497 (paper TWS), 4002 (paper Gateway), etc.
CLIENT_ID = 66       # Unique client ID; change if conflict

# ------------------------------------------------
# Strategy Settings
# ------------------------------------------------
SYMBOL = "BTC"
EXCHANGE = "PAXOS"
CURRENCY = "USD"

INITIAL_CAPITAL = 100.0
SHORT_SMA = 5
LONG_SMA = 15
VOLATILITY_LOOKBACK = 60
RISK_TOLERANCE = 0.02
MARGIN_BUFFER = 0.9

capital = INITIAL_CAPITAL
positions = {SYMBOL: 0.0}

# We'll store the live bars in a global DataFrame for logging
# This time we want it cumulative (keep old bars + new bars).
bars_df = pd.DataFrame()

ib = IB()

def connect_ib():
    """Connect to IBKR synchronously, logging the steps."""
    logging.info(f"Connecting to IBKR at {HOST}:{PORT}, clientId={CLIENT_ID}")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(1)  # 1=Real-time data
    logging.info("Connected to IBKR successfully.")

def start_data_stream():
    """
    Request streaming bars for BTC/PAXOS (historical data + live updates).
    We'll attach on_bar_update() to handle new bars.
    """
    contract = Crypto(SYMBOL, EXCHANGE, CURRENCY)
    ib.qualifyContracts(contract)
    logging.info(f"Qualified contract: {contract}")

    # For demonstration, we request 5-second bars. Adjust as needed.
    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='5 secs',
        whatToShow='MIDPOINT',
        useRTH=False,
        formatDate=1,
        keepUpToDate=True
    )
    bars.updateEvent += on_bar_update
    logging.info("Requested historical+streaming '5 secs' bars for BTC/PAXOS.")

def on_bar_update(bars, hasNewBar):
    """
    Callback for each new bar. We'll log:
    - Latest bar info (OHLC, volume)
    - Indicators (SMA, Volatility, Z-score)
    - Signal check (BUY/SELL/no signal)
    - Post-trade capital/position if a trade occurs
    """
    global bars_df

    if not hasNewBar:
        return

    # 1) Convert current bars to a new DataFrame
    new_df = util.df(bars).copy()  # copy() just to be safe

    # 2) Merge it with our global bars_df cumulatively
    if bars_df.empty:
        # First time: just take the entire new_df
        bars_df = new_df
    else:
        #print(bars_df)
        bars_df = pd.concat([bars_df, new_df]).drop_duplicates(subset='date').sort_values(by='date')

    # 3) The latest bar is now bars_df.iloc[-1] (last row)
    latest_bar = bars_df.iloc[-1]

    # 4) Log the new bar data
    logging.info(
        f"New Bar => time={latest_bar['date']}, "
        f"open={latest_bar['open']:.2f}, high={latest_bar['high']:.2f}, "
        f"low={latest_bar['low']:.2f}, close={latest_bar['close']:.2f}, vol={latest_bar['volume']:.4f}"
    )

    # 5) Enough bars for indicators?
    if len(bars_df) < max(LONG_SMA, VOLATILITY_LOOKBACK):
        logging.info(f"Not enough bars ({len(bars_df)}) for full indicator calc; need at least {max(LONG_SMA, VOLATILITY_LOOKBACK)}.")
        return

    # 6) Compute indicators on entire bars_df
    #    We'll do it on a copy or just do it in place
    df = bars_df.copy()
    df["SMA_Short"] = df["close"].rolling(SHORT_SMA).mean()
    df["SMA_Long"]  = df["close"].rolling(LONG_SMA).mean()
    df['Volatility'] = df['close'].pct_change().rolling(VOLATILITY_LOOKBACK).std().clip(lower=1e-4)
    df["Z_Score"] = (
        (df["close"] - df["close"].rolling(VOLATILITY_LOOKBACK).mean())
        / df["Volatility"]
    )
    df.dropna(inplace=True)

    if len(df) < LONG_SMA:
        logging.info("Still not enough bars after rolling indicators. Skipping.")
        return

    # The latest row with indicators
    row_latest = df.iloc[-1]
    short_now = row_latest["SMA_Short"]
    long_now  = row_latest["SMA_Long"]
    vol_now   = row_latest["Volatility"]
    z_now     = row_latest["Z_Score"]
    close_now = row_latest["close"]

    # Print some stats about indicators
    sma_diff = short_now - long_now
    logging.info(
        f"Indicators => shortSMA={short_now:.2f}, longSMA={long_now:.2f}, "
        f"smaDiff={sma_diff:.2f}, vol={vol_now:.5f}, Z={z_now:.2f}"
    )

    # Check for BUY/SELL signal
    signal = check_trade_signal(df)
    if not signal:
        logging.info("No signal triggered.")
        return

    # If we do have a signal, place a trade
    execute_trade(signal, close_now)

def check_trade_signal(df):
    """SMA crossover + Z-score logic. Return 'BUY', 'SELL', or None."""
    short_prev = df.iloc[-2]["SMA_Short"]
    long_prev  = df.iloc[-2]["SMA_Long"]
    short_now  = df.iloc[-1]["SMA_Short"]
    long_now   = df.iloc[-1]["SMA_Long"]
    z_now      = df.iloc[-1]["Z_Score"]

    # BUY: short crosses above long + Z < -1
    if short_prev < long_prev and short_now > long_now and z_now < -1:
        logging.info("SIGNAL => BUY")
        return "BUY"

    # SELL: short crosses below long + Z > 1
    if short_prev > long_prev and short_now < long_now and z_now > 1:
        logging.info("SIGNAL => SELL")
        return "SELL"

    return None

def execute_trade(order_type, price):
    """
    Attempt to place a MarketOrder for BTC. We log everything:
    - Chosen position size
    - 'IOC' time-in-force (if needed)
    - Post-trade capital/positions
    """
    global capital, positions

    vol_approx = np.std(list(positions.values()))
    units = risk_based_position_sizing(price, vol_approx)
    units = round(units, 8)  # 8 decimals for BTC

    if units <= 0:
        logging.info(f"Position size=0 => Skipping {order_type}.")
        return

    total_cost = units * price
    if order_type == "BUY" and capital < total_cost:
        logging.info(f"Not enough capital to BUY {units} BTC at ${price:.2f} => total cost ${total_cost:.2f}.")
        return

    contract = Crypto(SYMBOL, EXCHANGE, CURRENCY)
    order = MarketOrder(order_type, units)
    order.tif = "IOC"  # immediate or cancel

    logging.info(f"Placing {order_type} order => {units} BTC at ~${price:.2f}, totalCost ~${total_cost:.2f}")

    trade = ib.placeOrder(contract, order)

    def tradeCallback(trade_, fill_):
        logging.info(
            f"Order update => status={trade_.orderStatus.status}, "
            f"filled={trade_.orderStatus.filled}, "
            f"remaining={trade_.orderStatus.remaining}, "
            f"avgFillPrice={trade_.orderStatus.avgFillPrice}"
        )
    trade.fillEvent += tradeCallback

    # Update capital/position optimistically
    if order_type == "BUY":
        capital -= total_cost
        positions[SYMBOL] += units
    else:
        capital += total_cost
        positions[SYMBOL] -= units

    logging.info(
        f"EXECUTE => {order_type} {units} BTC at ${price:.2f}, "
        f"New capital=${capital:.2f}, New position={positions[SYMBOL]:.4f} BTC"
    )

def risk_based_position_sizing(price, volatility):
    """
    Simple formula => (capital * MARGIN_BUFFER * RISK_TOLERANCE) / (volatility * price).
    If volatility=0 => fallback 0.001 BTC
    """
    global capital
    if volatility > 0:
        risk_per_unit = volatility * price
        max_units = (capital * MARGIN_BUFFER * RISK_TOLERANCE) / risk_per_unit
        return max_units if max_units > 0 else 0
    return 0.001

def main():
    try:
        connect_ib()
        start_data_stream()

        logging.info("Starting ib_insync event loop (Ctrl+C to stop).")
        ib.run()  # block here until user stops or exception
    except Exception as ex:
        logging.error(f"Main caught exception: {ex}")

if __name__ == "__main__":
    main()