import time
import pandas as pd
from config import ALPACA_KEY, ALPACA_SECRET
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
# from alpaca.trading.enums import OrderSide, TimeInForce
# import matplotlib.pyplot as plt
from alpaca.trading.client import TradingClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
# from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import TrailingStopOrderRequest, LimitOrderRequest, MarketOrderRequest
from AppKit import NSApplication
from Foundation import NSObject
from datetime import datetime, timedelta
import mplfinance as mpf
import numpy as np


class AppDelegate(NSObject):
    def applicationSupportsSecureRestorableState_(self, app):
        return True


def submit_order(order_flag, symbol):
    trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
    # account = dict(trading_client.get_account())
    # for k, v in account.items():
    #     print(f"{k:30}{v}")

    # Get our position in AAPL.
    # stock_position = trading_client.get_open_position('SPY')

    # print(stock_position)

    orderside = None

    if order_flag == 1:
        orderside = 'buy'
    elif order_flag == -1:
        orderside = 'sell'

    order_details = MarketOrderRequest(
        symbol=symbol,
        qty=1,   # Quantity of Dollars
        side=orderside,
        time_in_force='day'
    )

    print(order_details)
    print()

    # client.submit_order(order_data=order_details)
    # trades = TradingStream(ALPACA_KEY, ALPACA_SECRET, paper=True)
    # async def trades_status(data):
    #     print(data)
    #
    # trades.subscribe_trade_updates(trades_status)
    # trades.run()


def fetch_market_data(symbol, start_date, end_date):

    # Replace 'your_api_endpoint' with the actual REST API endpoint for market data
    # data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    client = CryptoHistoricalDataClient()

    # time_frame_interval = TimeFrame(1, TimeFrameUnit.Minute)
    # time_frame_interval = TimeFrame(5, TimeFrameUnit.Minute)
    # time_frame_interval = TimeFrame(15, TimeFrameUnit.Minute)
    time_frame_interval = TimeFrame(1, TimeFrameUnit.Day)
    # time_frame_interval = TimeFrame(30, TimeFrameUnit.Minute)

    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=time_frame_interval,
        # limit=900,
        start=start_date,
        end=end_date,

    )

    # print(request_params)

    crypto_bars = client.get_crypto_bars(request_params).df.tz_convert('America/New_York', level=1)

    # Use between_time on the DataFrame
    time_from = '8:00'
    time_to = '16:00'

    # Convert the index to DatetimeIndex and set the timezone to 'America/New_York' (EST)
    crypto_bars.index = pd.to_datetime(crypto_bars.index.get_level_values(1), utc=True).tz_convert('America/New_York')

    # crypto_bars = crypto_bars.between_time(time_from, time_to)

    return crypto_bars


def calculate_ema_np(data, column, window):

    # Calculate EMA using numpy
    ema = pd.Series(data[column]).ewm(span=window, adjust=False).mean().values
    return ema


def write_signals_file(message, file_path):

    with open(file_path, "w") as file:
        file.write(message + "\n")


def write_orders_file(message, file_path):

    with open(file_path, "a") as file:
        file.write(message + "\n")


def write_timeseries_file(message, file_path):

    with open(file_path, "w") as file:
        file.write(message + "\n")


def read_orders_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Assuming each line in the file is a new record and contains the timestamp
            return {line.split(',')[0] for line in file}  # Extracting the timestamp part
    except FileNotFoundError:
        return set()


def generate_ema_signals(stock_data, symbol):

    stock_data['ema_5'] = calculate_ema_np(stock_data, 'close', 5)
    stock_data['ema_8'] = calculate_ema_np(stock_data, 'close', 8)
    stock_data['ema_13'] = calculate_ema_np(stock_data, 'close', 13)

    # Initialize if an order was place on the instance of the dataframe
    # orders_placed = {}

    # Generate signals based on EMA crossovers adding a column 'Signal' and setting it to '0'
    stock_data['signal'] = 0

    # Buy signal conditions
    stock_data.loc[(stock_data['ema_5'] > stock_data['ema_8']) & (stock_data['ema_8'] > stock_data['ema_13']),
                   'signal'] = 1

    # Sell signal conditions
    stock_data.loc[(stock_data['ema_5'] < stock_data['ema_8']) & (stock_data['ema_8'] < stock_data['ema_13']),
                   'signal'] = -1

    signal_changes = stock_data['signal'].diff().ne(0) & stock_data['signal'].ne(0)

    message_data = f"{stock_data.to_string()}"
    write_timeseries_file(message_data, 'ema_time_series.txt')

    # Read the existing orders
    existing_trades = read_orders_file('ema_orders.txt')

    # Assuming signal_changes is a Pandas Series with -1, 0, 1 values
    if signal_changes.any():
        last_signal_change_index = signal_changes[signal_changes].index[-1]
        filtered_stock_data = stock_data.loc[last_signal_change_index]
        signal_value = filtered_stock_data['signal']
        timestamp = filtered_stock_data.name  # Replace 'timestamp' with the actual column name
        close = filtered_stock_data['close']

        # Convert the Timestamp object to a string with the desired format
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
        # Replace '-0500' with '-05:00' for time zone offset
        timestamp_str = timestamp_str[:-2] + ':' + timestamp_str[-2:]

        if signal_value in [1, -1] and timestamp_str not in existing_trades:

            submit_order(signal_value, symbol)

            message = f"{timestamp}, Close: {close}, Signal Value: {signal_value} Order Submitted"
            write_orders_file(message, 'ema_orders.txt')
            print(message + "\n")

        else:
            print(f"{timestamp}, No order has been submitted...rest easy \n")

    return stock_data


def plot_chart(stock_data, symbol):

    stock_data['timestamp'] = stock_data.index.get_level_values('timestamp').tolist()

    stock_data.set_index('timestamp', inplace=True)  # Set the index to 'timestamp'

    # Assuming stock_data['signal'] and mask are already defined
    mask = stock_data['signal'] != 0

    # Apply diff() only on rows where signal is not zero and merge it back
    stock_data['diff_values'] = stock_data.loc[mask, 'signal'].diff().reindex(stock_data.index)

    # Identify buy signals (diff_values == 2) and extract timestamps and close prices
    buy_signals = stock_data[stock_data['diff_values'] == 2]['close']

    # Identify sell signals (diff_values == -2) and extract timestamps and close prices
    sell_signals = stock_data[stock_data['diff_values'] == -2]['close']

    message = f"BUY {buy_signals.to_string()} \nSELL {sell_signals.to_string()} \n"
    write_signals_file(message, 'ema_signals.txt')

    buy_signals = buy_signals.reindex(stock_data.index, fill_value=None)
    sell_signals = sell_signals.reindex(stock_data.index, fill_value=None)

    # Visually off sets the markers
    visual_offset = 4000

    buy_signals = buy_signals - visual_offset  # Decrease y-value for visual effect
    sell_signals = sell_signals + visual_offset  # Increase y-value for visual effect

    # Now create the mplfinance plots for buy and sell signals with an offset on the graph
    buy_marker = mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='green', alpha=0.5)
    sell_marker = mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='red', alpha=0.5)

    # Assuming 'my_style' is already defined and 'symbol' is a string with the stock symbol
    my_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 8})

    # Add the buy and sell markers to the existing plot
    mpf.plot(stock_data, type='candle', style=my_style,
             title=f'EMA Crossover Momentum Trading Strategy - {symbol}',
             ylabel='Price',
             figsize=(10, 6),
             addplot=[
                 mpf.make_addplot(stock_data['ema_5'], color='lightblue'),
                 mpf.make_addplot(stock_data['ema_8'], color='royalblue'),
                 mpf.make_addplot(stock_data['ema_13'], color='navy'),
                 buy_marker,
                 sell_marker
             ])

    mpf.show()


def main():

    # Example usage
    # symbol = 'QQQ'    # Replace with the desired stock symbol
    symbol = 'BTC/USD'    # Replace with the desired stock symbol
    delay_seconds = 50
    # delay_seconds = 30

    while True:

        today = datetime.now()
        start_date = today - timedelta(days=90)
        # start_date = today
        start_date = start_date.strftime("%Y-%m-%d")
        start_date = pd.to_datetime(start_date).tz_localize('America/New_York')
        end_date = today
        end_date = pd.to_datetime(end_date).tz_localize('America/New_York')

        # Fetch market data using the business day date range
        stock_data = fetch_market_data(symbol, start_date, end_date)

        # Generate EMA signals with a 5, 8, and 13 timeframe window
        generate_ema_signals(stock_data, symbol)

        # Plot Candlestick Chart
        plot_chart(stock_data, symbol)

        # Sleep
        time.sleep(delay_seconds)


if __name__ == "__main__":

    delegate = AppDelegate.alloc().init()
    app = NSApplication.sharedApplication()
    app.setDelegate_(delegate)

    # Call your main function
    main()

    # Start the application event loop
    app.run()
