"""
sync_dash_demo.py
Demonstrates a single-threaded approach where ib_insync and Dash run together
thanks to `nest_asyncio`.
We display a simple "live table" of the latest bars, updated every 5s.
"""

import time
import datetime
import pandas as pd

import nest_asyncio
nest_asyncio.apply()  # Allows re-entrant asyncio loops in one thread

from ib_insync import IB, Crypto
from dash import Dash, dcc, html, dash_table, Input, Output

# -------------------------------------------------
# 1) IBKR Connection & Setup
# -------------------------------------------------
ib = IB()

HOST = '127.0.0.1'
PORT = 4001  # or 4001/4002 etc. Make sure it matches TWS/IB Gateway
CLIENT_ID = 1234

def connect_ib():
    print(f"[{datetime.datetime.now()}] Connecting to IBKR...")
    # This will block until connected or times out
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"[{datetime.datetime.now()}] Connected!")

    # For real-time data (if you have subscription):
    ib.reqMarketDataType(1)

# -------------------------------------------------
# 2) Request Streaming Bars (1-second or 5-second bars)
#    But we'll do "keepUpToDate" = True historical data for a simpler approach
# -------------------------------------------------

bars_df = pd.DataFrame()

def start_data_stream():
    contract = Crypto('BTC', 'PAXOS', 'USD')
    ib.qualifyContracts(contract)

    # We'll request 1 day of 1-min bars, then keepUpToDate = True for streaming
    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=False,
        formatDate=1,
        keepUpToDate=True
    )
    bars.updateEvent += on_bar_update


def on_bar_update(bars, hasNewBar):
    global bars_df
    if not hasNewBar:
        return
    df = bars.df()
    # you could compute strategy signals here if you want
    bars_df = df  # store latest data in global

# -------------------------------------------------
# 3) Dash App
# -------------------------------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Synchronous Dash + IBKR Demo"),
    html.Div("Displays the last ~50 bars for BTC/PAXOS in a table. Refreshes every 5s."),

    # The Dash DataTable
    dash_table.DataTable(
        id='bars-table',
        columns=[
            {'name': 'DateTime', 'id': 'date'},
            {'name': 'Open', 'id': 'open'},
            {'name': 'High', 'id': 'high'},
            {'name': 'Low', 'id': 'low'},
            {'name': 'Close', 'id': 'close'},
            {'name': 'Volume', 'id': 'volume'}
        ],
        data=[]
    ),
    dcc.Interval(id='interval', interval=5*1000, n_intervals=0)
])

@app.callback(
    Output('bars-table', 'data'),
    [Input('interval', 'n_intervals')]
)
def update_table(n):
    """
    Called every 5s. We'll return the latest ~50 bars from bars_df
    in a list-of-dicts format for the Dash table.
    """
    global bars_df
    if bars_df.empty:
        return []

    # Convert the last 50 bars to a list of dicts
    df_tail = bars_df.tail(50).copy()
    # dash_table can't handle datetime as index easily, so let's reset the index
    df_tail = df_tail.reset_index()
    df_tail.rename(columns={'date': 'time'}, inplace=True)

    # Build records
    records = []
    for _, row in df_tail.iterrows():
        records.append({
            'date': str(row['date']),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
        })
    return records

# -------------------------------------------------
# 4) Single-thread "main"
# -------------------------------------------------
def main():
    connect_ib()         # synchronous connect
    start_data_stream()  # start streaming bars

    # Now we run the Dash server in the same thread
    # Because we used nest_asyncio, ib_insync's event loop won't conflict.
    print(f"[{datetime.datetime.now()}] Starting Dash server on http://127.0.0.1:8050")
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
