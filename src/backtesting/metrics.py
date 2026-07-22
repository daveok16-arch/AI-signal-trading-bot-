"""Performance metrics for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    returns: pd.Series = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    
    if returns is None and equity_curve is not None:
        returns = equity_curve.pct_change().dropna()
    
    if returns is None or len(returns) == 0:
        return {}
    
    metrics = {}
    
    # Basic returns
    metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    metrics['annualized_return'] = _annualize_return(returns)
    
    # Risk metrics
    metrics['volatility'] = returns.std() * np.sqrt(252 * 1440)  # Annualized for 1min
    metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(252 * 1440)
    
    # Sharpe and Sortino
    excess_returns = returns - risk_free_rate / (252 * 1440)
    metrics['sharpe_ratio'] = excess_returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 1440)
    metrics['sortino_ratio'] = excess_returns.mean() / (metrics['downside_volatility'] / np.sqrt(252 * 1440) + 1e-10) * np.sqrt(252 * 1440)
    
    # Drawdown
    drawdown_series = _calculate_drawdown(equity_curve)
    metrics['max_drawdown'] = drawdown_series.min()
    metrics['max_drawdown_duration'] = _max_drawdown_duration(drawdown_series)
    metrics['avg_drawdown'] = drawdown_series[drawdown_series < 0].mean()
    
    # Calmar ratio
    metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
    
    # Trade metrics
    if trades is not None and len(trades) > 0:
        trade_metrics = _calculate_trade_metrics(trades)
        metrics.update(trade_metrics)
    
    # Risk metrics
    metrics['var_95'] = returns.quantile(0.05)
    metrics['var_99'] = returns.quantile(0.01)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
    
    # Tail ratio
    tail_5 = returns.quantile(0.95)
    tail_95 = returns.quantile(0.05)
    metrics['tail_ratio'] = tail_5 / abs(tail_95) if tail_95 != 0 else 0
    
    # Gain to Pain ratio
    gains = returns[returns > 0].sum()
    pains = abs(returns[returns < 0].sum())
    metrics['gain_to_pain_ratio'] = gains / pains if pains > 0 else 0
    
    # Recovery factor
    metrics['recovery_factor'] = metrics['total_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
    
    # Ulcer index
    metrics['ulcer_index'] = np.sqrt((drawdown_series ** 2).mean())
    
    # Omega ratio
    metrics['omega_ratio'] = _omega_ratio(returns)
    
    return metrics


def _calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate maximum drawdown duration in periods."""
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0
    
    drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    durations = in_drawdown.groupby(drawdown_groups).sum()
    return int(durations.max())


def _annualize_return(returns: pd.Series) -> float:
    """Annualize returns for 1-minute data."""
    n_periods = len(returns)
    if n_periods == 0:
        return 0
    total_return = (1 + returns).prod() - 1
    years = n_periods / (252 * 1440)
    if years > 0:
        return (1 + total_return) ** (1 / years) - 1
    return total_return


def _omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    """Calculate Omega ratio."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    return gains / losses if losses > 0 else 0


def _calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """Calculate trade-based metrics."""
    metrics = {}
    
    metrics['n_trades'] = len(trades)
    
    winning = trades[trades['pnl'] > 0]
    losing = trades[trades['pnl'] < 0]
    
    metrics['win_rate'] = len(winning) / len(trades) if len(trades) > 0 else 0
    metrics['loss_rate'] = len(losing) / len(trades) if len(trades) > 0 else 0
    
    metrics['avg_win'] = winning['pnl'].mean() if len(winning) > 0 else 0
    metrics['avg_loss'] = losing['pnl'].mean() if len(losing) > 0 else 0
    
    metrics['profit_factor'] = abs(winning['pnl'].sum() / losing['pnl'].sum()) if losing['pnl'].sum() != 0 else np.inf
    
    metrics['expectancy'] = metrics['win_rate'] * metrics['avg_win'] + metrics['loss_rate'] * metrics['avg_loss']
    
    metrics['avg_trade'] = trades['pnl'].mean()
    
    if metrics['avg_loss'] != 0:
        metrics['risk_reward_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['risk_reward_ratio'] = 0
    
    # Consecutive wins/losses
    trades['win'] = trades['pnl'] > 0
    metrics['max_consecutive_wins'] = _max_consecutive(trades['win'], True)
    metrics['max_consecutive_losses'] = _max_consecutive(trades['win'], False)
    
    # Average holding time
    if 'holding_minutes' in trades.columns:
        metrics['avg_holding_minutes'] = trades['holding_minutes'].mean()
    
    # Exit reasons
    if 'exit_reason' in trades.columns:
        metrics['exit_reasons'] = trades['exit_reason'].value_counts().to_dict()
    
    return metrics


def _max_consecutive(series: pd.Series, value: bool) -> int:
    """Calculate maximum consecutive occurrences."""
    max_count = 0
    current = 0
    for v in series:
        if v == value:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def generate_performance_summary(metrics: Dict[str, float]) -> str:
    """Generate human-readable performance summary."""
    lines = [
        "=" * 50,
        "PERFORMANCE SUMMARY",
        "=" * 50,
        f"Total Return:        {metrics.get('total_return', 0):.2%}",
        f"Annualized Return:   {metrics.get('annualized_return', 0):.2%}",
        f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}",
        f"Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}",
        f"Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}",
        f"Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}",
        f"Volatility:          {metrics.get('volatility', 0):.2%}",
        "",
        "TRADE METRICS",
        "-" * 30,
        f"Total Trades:        {metrics.get('n_trades', 0)}",
        f"Win Rate:            {metrics.get('win_rate', 0):.2%}",
        f"Profit Factor:       {metrics.get('profit_factor', 0):.2f}",
        f"Expectancy:          {metrics.get('expectancy', 0):.4f}",
        f"Avg Win:             {metrics.get('avg_win', 0):.2f}",
        f"Avg Loss:            {metrics.get('avg_loss', 0):.2f}",
        f"Risk/Reward:         {metrics.get('risk_reward_ratio', 0):.2f}",
        f"Max Consec Wins:     {metrics.get('max_consecutive_wins', 0)}",
        f"Max Consec Losses:   {metrics.get('max_consecutive_losses', 0)}",
        "",
        "RISK METRICS",
        "-" * 30,
        f"VaR 95%:             {metrics.get('var_95', 0):.4f}",
        f"CVaR 95%:            {metrics.get('cvar_95', 0):.4f}",
        f"Tail Ratio:          {metrics.get('tail_ratio', 0):.2f}",
        f"Gain/Pain Ratio:     {metrics.get('gain_to_pain_ratio', 0):.2f}",
        f"Ulcer Index:         {metrics.get('ulcer_index', 0):.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    n = 10000
    returns = pd.Series(np.random.randn(n) * 0.001 + 0.0001)
    equity = (1 + returns).cumprod() * 100000
    
    metrics = calculate_metrics(equity, returns=returns)
    print(generate_performance_summary(metrics))
