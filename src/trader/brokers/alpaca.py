from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.requests import GetOrdersRequest
from requests.exceptions import RequestException
from loguru import logger
from datetime import datetime, timezone

class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
    def get_positions(self):
        try:
            return self.trading_client.get_all_positions()
        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (positions): {e}")

    def get_today_fills(self):
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            request = GetOrdersRequest(
                status="closed",
                after=today,
                direction="asc",
            )
            return self.trading_client.get_orders(request)
        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (orders): {e}")
    def get_account(self):
        try:
            return self.trading_client.get_account()
        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (account): {e}")

    def get_last_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self.data_client.get_stock_latest_trade(request)
        trade = trades[symbol]
        return float(trade.price)

    def place_market_order(self, symbol: str, side: str, qty: int):
        logger.info(f"Placing {side.upper()} order for {qty} {symbol}")

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return self.trading_client.submit_order(order)
