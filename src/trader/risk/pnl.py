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
    try:
        positions = broker.get_positions()
        orders = broker.get_today_fills()
    except RuntimeError as e:
        logger.error(f"PNL unavailable: {e}")
        return 0.0  # Fail SAFE: do not trade blindly

    unrealized = sum(float(p.unrealized_pl) for p in positions)
    realized = 0.0

    for order in orders:
        if order.filled_avg_price and order.filled_qty:
            direction = -1 if order.side == "buy" else 1
            realized += direction * float(order.filled_qty) * float(order.filled_avg_price)

    total = unrealized + realized

    logger.info(
        f"PNL | unrealized={unrealized:.2f} realized={realized:.2f} total={total:.2f}"
    )

    return total
