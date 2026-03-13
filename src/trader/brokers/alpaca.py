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
from datetime import datetime, timezone

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

    def wait_for_fill(self, order_id: str, timeout_s: float = 30.0, poll_s: float = 1.5):
        """
        Polls an order until it is filled or a timeout occurs.
        Returns a dict: {filled: bool, order: object | None}
        """
        if not order_id:
            return {"filled": False, "order": None}

        deadline = time.time() + float(timeout_s)
        last_order = None

        while time.time() < deadline:
            try:
                o = self.trading_client.get_order_by_id(order_id)
                last_order = o
                status_raw = getattr(o, "status", "")
                status = str(status_raw).lower()
                filled_qty = float(getattr(o, "filled_qty", 0) or 0)
                is_filled = ("filled" in status) or ("closed" in status)
                if is_filled and filled_qty > 0:
                    return {"filled": True, "order": o}
                if status in {"canceled", "rejected", "expired"}:
                    return {"filled": False, "order": o}
            except Exception as e:
                logger.warning(f"Order fetch failed {order_id}: {e}")

            time.sleep(float(poll_s))

        return {"filled": False, "order": last_order}
    
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

    def cancel_stale_open_orders(self, max_age_minutes: int = 60) -> int:
        """
        Cancel open orders older than max_age_minutes. Returns count canceled.
        """
        try:
            request = GetOrdersRequest(status="open")
            orders = self.trading_client.get_orders(request)
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return 0

        canceled = 0
        now_utc = datetime.now(timezone.utc)
        for o in orders:
            created_at = getattr(o, "created_at", None)
            if created_at is None:
                continue
            try:
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age_min = (now_utc - created_at).total_seconds() / 60.0
            except Exception:
                continue
            if age_min >= float(max_age_minutes):
                try:
                    self.trading_client.cancel_order_by_id(o.id)
                    canceled += 1
                    logger.warning(
                        f"Cancelled stale order {o.id} "
                        f"{getattr(o, 'symbol', '')} age={age_min:.1f}m"
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel order {o.id}: {e}")
        return canceled
 
