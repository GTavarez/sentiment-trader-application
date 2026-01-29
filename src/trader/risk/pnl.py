from loguru import logger


def calculate_unrealized_pnl(positions) -> float:
    pnl = 0.0
    for pos in positions:
        pnl += float(pos.unrealized_pl)
    return pnl


def calculate_realized_pnl(orders) -> float:
    pnl = 0.0
    for order in orders:
        if order.filled_avg_price and order.filled_qty:
            # BUY reduces cash, SELL increases cash
            direction = -1 if order.side == "buy" else 1
            pnl += direction * float(order.filled_qty) * float(order.filled_avg_price)
    return pnl
    

def calculate_daily_pnl(broker) -> float:
    """
    Calculates today's realized PnL from Alpaca fills.
    Safe + deterministic.
    """

    try:
        orders = broker.get_today_fills()
    except Exception as e:
        logger.error(f"Failed to fetch today's fills: {e}")
        return 0.0

    realized = 0.0

    for order in orders:
        if not order.filled_qty or not order.filled_avg_price:
            continue

        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)

        if order.side.lower() == "buy":
            realized -= qty * price
        elif order.side.lower() == "sell":
            realized += qty * price

    logger.info(
        f"PNL | realized={realized:.2f} unrealized=0.00 total={realized:.2f}"
    )

    return realized
