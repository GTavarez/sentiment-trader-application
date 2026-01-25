from dataclasses import dataclass
from loguru import logger


@dataclass
class RiskLimits:
    max_trades_per_day: int
    max_position_usd: float
    daily_loss_limit_usd: float


class RiskManager:
    """
    MVP Risk Manager.
    - caps trades/day
    - caps position notional per trade
    - kill switch on daily loss
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def allow_trade(self, state: dict, proposed: dict) -> tuple[bool, str]:
        """
        state: {
          "trades_today": int,
          "pnl_today_usd": float
        }
        proposed: {
          "symbol": str,
          "side": "buy"|"sell",
          "qty": int,
          "price": float,
          "notional_usd": float
        }
        """
        if state["pnl_today_usd"] <= -self.limits.daily_loss_limit_usd:
            return False, "KILL_SWITCH: daily loss limit hit"

        if state["trades_today"] >= self.limits.max_trades_per_day:
            return False, "max trades/day reached"

        if proposed["notional_usd"] > self.limits.max_position_usd:
            return False, "position size exceeds max_position_usd"

        if proposed["qty"] <= 0:
            return False, "qty must be >= 1"

        return True, "ok"

    def log_block(self, proposed: dict, reason: str) -> None:
        logger.warning(
            f"BLOCKED trade {proposed['side'].upper()} {proposed['qty']} {proposed['symbol']} "
            f"notional=${proposed['notional_usd']:.2f} reason={reason}"
        )
