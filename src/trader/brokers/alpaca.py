from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.requests import GetOrdersRequest
from requests.exceptions import RequestException
from loguru import logger
from alpaca.common.exceptions import APIError
import time

class AlpacaBroker:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        last_price_max_retries: int = 3,
        last_price_retry_base_s: float = 0.5,
        last_price_retry_max_s: float = 5.0,
    ):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self.last_price_max_retries = max(1, int(last_price_max_retries))
        self.last_price_retry_base_s = max(0.0, float(last_price_retry_base_s))
        self.last_price_retry_max_s = max(0.0, float(last_price_retry_max_s))
    def get_positions(self):
        try:
            return self.trading_client.get_all_positions()
        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (positions): {e}")

    def get_today_fills(self):
        """
        Return ONLY today's filled orders (UTC day).
        """
        try:
            request = GetOrdersRequest(
                status="closed",
                direction="asc",
            )

            orders = self.trading_client.get_orders(request)

            todays = []
            for o in orders:
                filled_at = getattr(o, "filled_at", None)
                if not filled_at:
                    continue

            return todays

        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (orders): {e}") from e

    def get_account(self):
        try:
            return self.trading_client.get_account()
        except RequestException as e:
            raise RuntimeError(f"Alpaca connection failed (account): {e}")

    def get_last_price(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        last_err: Exception | None = None
        for attempt in range(1, self.last_price_max_retries + 1):
            try:
                trades = self.data_client.get_stock_latest_trade(request)
                trade = trades[symbol]
                return float(trade.price)
            except (RequestException, APIError, Exception) as e:
                last_err = e
                if attempt >= self.last_price_max_retries:
                    break
                sleep_s = min(
                    self.last_price_retry_base_s * (2 ** (attempt - 1)),
                    self.last_price_retry_max_s,
                )
                logger.warning(
                    f"Latest price fetch failed for {symbol} "
                    f"(attempt {attempt}/{self.last_price_max_retries}): {e}"
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise RuntimeError(f"Failed to fetch latest price for {symbol}: {last_err}")

    def place_market_order(self, symbol: str, side: str, qty: int):
        logger.info(f"Placing {side.upper()} order for {qty} {symbol}")

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return self.trading_client.submit_order(order)
    
    def get_position_qty(self, symbol: str) -> int:
        try:
           positions = self.trading_client.get_all_positions()
           for p in positions:
               if p.symbol.upper() == symbol.upper():
                  return int(p.qty)
           return 0
        except Exception as e:
              logger.error(f"Position lookup failed for {symbol}: {e}")
              return 0

    def get_position(self, symbol: str):
        try:
            positions = self.trading_client.get_all_positions()
            for p in positions:
                if p.symbol.upper() == symbol.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"Position lookup failed for {symbol}: {e}")
            return None

    def cancel_open_orders(self, symbol: str):
        try:
           request = GetOrdersRequest(status="open")
           orders = self.trading_client.get_orders(request)

           for o in orders:
               if o.symbol == symbol:
                logger.warning(f"Cancelling open order {o.id} for {symbol}")
                self.trading_client.cancel_order_by_id(o.id)

        except APIError as e:
             logger.error(f"Failed to cancel orders for {symbol}: {e}")
 
