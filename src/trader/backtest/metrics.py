import pandas as pd
import numpy as np

def compute_metrics(equity_df: pd.DataFrame, freq: int = 252):
    """
    Compute performance metrics from equity curve.
    Expects columns: [timestamp, equity]
    """

    if equity_df.empty or "equity" not in equity_df.columns:
        return {}

    df = equity_df.copy().sort_values("timestamp")

    # Returns
    df["returns"] = df["equity"].pct_change().fillna(0)

    # Sharpe Ratio
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()

    sharpe = (
        (mean_ret / std_ret) * np.sqrt(freq)
        if std_ret > 0 else 0.0
    )

    # Drawdown
    cum_max = df["equity"].cummax()
    drawdown = (df["equity"] - cum_max) / cum_max
    max_drawdown = drawdown.min()

    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1

    return {
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(abs(max_drawdown) * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "trades": len(df),
    }
