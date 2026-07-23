"""Reporting module for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .metrics import calculate_metrics

logger = __import__('logging').getLogger(__name__)


def generate_report(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    returns: pd.Series = None,
    benchmark: pd.Series = None,
) -> str:
    """Generate performance report as markdown string."""
    metrics = calculate_metrics(equity_curve, trades, returns)
    
    lines = [
        "# Performance Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]
    
    if 'total_return' in metrics:
        lines.append(f"- **Total Return:** {metrics['total_return']:.2%}")
    if 'annualized_return' in metrics:
        lines.append(f"- **Annualized Return:** {metrics['annualized_return']:.2%}")
    if 'sharpe_ratio' in metrics:
        lines.append(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}")
    if 'sortino_ratio' in metrics:
        lines.append(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.2f}")
    if 'max_drawdown' in metrics:
        lines.append(f"- **Max Drawdown:** {metrics['max_drawdown']:.2%}")
    if 'n_trades' in metrics:
        lines.append(f"- **Total Trades:** {metrics['n_trades']}")
    if 'win_rate' in metrics:
        lines.append(f"- **Win Rate:** {metrics['win_rate']:.2%}")
    if 'profit_factor' in metrics:
        lines.append(f"- **Profit Factor:** {metrics['profit_factor']:.2f}")
    
    if trades is not None and len(trades) > 0:
        lines.extend([
            "",
            "## Trade Analysis",
            "",
            f"- **Average Win:** ${metrics.get('avg_win', 0):.2f}",
            f"- **Average Loss:** ${metrics.get('avg_loss', 0):.2f}",
            f"- **Risk/Reward:** {metrics.get('risk_reward_ratio', 0):.2f}",
            f"- **Max Consecutive Wins:** {metrics.get('max_consecutive_wins', 0)}",
            f"- **Max Consecutive Losses:** {metrics.get('max_consecutive_losses', 0)}",
        ])
    
    if benchmark is not None:
        lines.extend([
            "",
            "## Benchmark Comparison",
            "",
        ])
        # Add benchmark comparison if available
    
    return "\n".join(lines)


def generate_html_report(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    returns: pd.Series = None,
) -> str:
    """Generate HTML performance report."""
    markdown = generate_report(equity_curve, trades, returns)
    import markdown as md
    return md.markdown(markdown)


if __name__ == "__main__":
    # Test with dummy data
    import numpy as np
    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000) * 0.01 + 0.0005)
    equity = (1 + returns).cumprod() * 100000
    
    report = generate_report(equity)
    print(report)
