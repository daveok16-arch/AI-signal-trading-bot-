"""Backtesting engine for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = 1
    SELL = -1
    WAIT = 0


@dataclass
class Trade:
    """Trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    max_favorable: float = 0
    max_adverse: float = 0
    holding_minutes: int = 0
    exit_reason: str = ""


@dataclass
class Position:
    """Current open position."""
    direction: int
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0
    max_price: float = 0
    min_price: float = 0


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        bt_config = self.config.get_section('backtest')
        
        self.initial_capital = bt_config.get('initial_capital', 100000)
        self.position_size = bt_config.get('position_size', 0.1)
        self.max_positions = bt_config.get('max_positions', 1)
        self.commission = bt_config.get('commission', 0.0001)
        self.slippage = bt_config.get('slippage', 0.0001)
        self.stop_loss_pct = bt_config.get('stop_loss_pct', 0.005)
        self.take_profit_pct = bt_config.get('take_profit_pct', 0.01)
        self.trailing_stop = bt_config.get('trailing_stop', True)
        self.trailing_activation = bt_config.get('trailing_activation', 0.005)
        self.trailing_distance = bt_config.get('trailing_distance', 0.003)
        self.max_holding_minutes = bt_config.get('max_holding_minutes', 15)
        self.max_daily_trades = bt_config.get('max_daily_trades', 50)
        self.max_daily_loss_pct = bt_config.get('max_daily_loss_pct', 0.02)
        self.max_drawdown_pct = bt_config.get('max_drawdown_pct', 0.15)
        self.risk_per_trade = bt_config.get('risk_per_trade', 0.01)
        self.position_sizing = bt_config.get('position_sizing', 'kelly')
        self.kelly_fraction = bt_config.get('kelly_fraction', 0.5)
        self.min_confidence = bt_config.get('min_confidence', 0.55)
        self.min_risk_reward = bt_config.get('min_risk_reward', 1.5)
        self.max_spread_pct = bt_config.get('max_spread_pct', 0.0005)
        
        # Session filter
        session_config = bt_config.get('session_filter', {})
        self.session_filter_enabled = session_config.get('enabled', True)
        self.allowed_sessions = session_config.get('allowed_sessions', ['london', 'new_york', 'overlap'])
        
        # State
        self.capital = self.initial_capital
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.open_positions: List[Position] = []
        self.daily_stats: Dict[str, Dict] = {}
        self.current_day: Optional[str] = None
        self.daily_trades = 0
        self.daily_pnl = 0
        
        # Results
        self.results: Dict[str, Any] = {}
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        confidence: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            df: OHLCV DataFrame
            signals: Signal series (1=buy, -1=sell, 0=wait)
            confidence: Confidence scores for signals
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest on {len(df)} bars")
        
        # Reset state
        self._reset()
        
        # Align data
        common_idx = df.index.intersection(signals.index)
        df = df.loc[common_idx]
        signals = signals.loc[common_idx]
        
        if confidence is not None:
            confidence = confidence.loc[common_idx]
        else:
            confidence = pd.Series(1.0, index=common_idx)
        
        # Iterate through bars
        for i in range(len(df)):
            timestamp = df.index[i]
            row = df.iloc[i]
            signal = signals.iloc[i]
            conf = confidence.iloc[i]
            
            # Check new day
            self._check_new_day(timestamp)
            
            # Update positions
            self._update_positions(row, timestamp)
            
            # Check risk limits
            if not self._check_risk_limits():
                continue
            
            # Process signals
            if signal != 0 and conf >= self.min_confidence:
                self._process_signal(signal, conf, row, timestamp)
            
            # Record equity
            self._record_equity(timestamp, row)
        
        # Close remaining positions
        self._close_all_positions(df.iloc[-1], df.index[-1], "end_of_backtest")
        
        # Calculate results
        self.results = self._calculate_results(df)
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}, Final equity: {self.capital:.2f}")
        
        return self.results
    
    def _reset(self) -> None:
        """Reset engine state."""
        self.capital = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.open_positions = []
        self.daily_stats = {}
        self.current_day = None
        self.daily_trades = 0
        self.daily_pnl = 0
    
    def _check_new_day(self, timestamp: pd.Timestamp) -> None:
        """Check if new trading day."""
        day = timestamp.strftime('%Y-%m-%d')
        if day != self.current_day:
            if self.current_day is not None:
                self.daily_stats[self.current_day] = {
                    'trades': self.daily_trades,
                    'pnl': self.daily_pnl,
                    'capital': self.capital,
                }
            self.current_day = day
            self.daily_trades = 0
            self.daily_pnl = 0
    
    def _check_risk_limits(self) -> bool:
        """Check risk limits."""
        # Max daily trades
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Max daily loss
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            return False
        
        # Max drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            drawdown = (peak - self.capital) / peak
            if drawdown > self.max_drawdown_pct:
                return False
        
        # Max positions
        if len(self.open_positions) >= self.max_positions:
            return False
        
        return True
    
    def _process_signal(
        self,
        signal: int,
        confidence: float,
        row: pd.Series,
        timestamp: pd.Timestamp,
    ) -> None:
        """Process trading signal."""
        # Check session filter
        if self.session_filter_enabled and not self._is_allowed_session(timestamp):
            return
        
        # Check spread
        spread_pct = (row['high'] - row['low']) / row['close']
        if spread_pct > self.max_spread_pct:
            return
        
        direction = signal  # 1 for buy, -1 for sell
        
        # Calculate position size
        size = self._calculate_position_size(row['close'], confidence)
        if size <= 0:
            return
        
        # Calculate stop loss and take profit
        if direction == 1:
            stop_loss = row['close'] * (1 - self.stop_loss_pct)
            take_profit = row['close'] * (1 + self.take_profit_pct)
        else:
            stop_loss = row['close'] * (1 + self.stop_loss_pct)
            take_profit = row['close'] * (1 - self.take_profit_pct)
        
        # Check risk/reward
        risk = abs(row['close'] - stop_loss)
        reward = abs(take_profit - row['close'])
        if risk > 0 and reward / risk < self.min_risk_reward:
            return
        
        # Apply slippage
        entry_price = row['close'] * (1 + self.slippage * direction)
        
        # Create position
        position = Position(
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=0,
            max_price=row['high'] if direction == 1 else row['low'],
            min_price=row['low'] if direction == 1 else row['high'],
        )
        
        self.open_positions.append(position)
        self.daily_trades += 1
        
        logger.debug(f"Opened {direction} position at {entry_price:.5f}, size={size:.2f}")
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size based on risk management."""
        if self.position_sizing == 'fixed':
            # Fixed percentage of capital
            risk_amount = self.capital * self.risk_per_trade
            size = risk_amount / (price * self.stop_loss_pct)
            
        elif self.position_sizing == 'kelly':
            # Kelly criterion (simplified)
            win_rate = 0.55  # Estimated
            avg_win = self.take_profit_pct
            avg_loss = self.stop_loss_pct
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = max(0, kelly) * self.kelly_fraction
            size = self.capital * kelly / price
            
        elif self.position_sizing == 'volatility':
            # Volatility-adjusted (placeholder)
            size = self.capital * self.position_size / price
            
        else:
            size = self.capital * self.position_size / price
        
        # Adjust by confidence
        size *= confidence
        
        # Apply max position size
        max_size = self.capital * self.position_size / price
        size = min(size, max_size)
        
        return size
    
    def _update_positions(self, row: pd.Series, timestamp: pd.Timestamp) -> None:
        """Update open positions with current bar."""
        positions_to_close = []
        
        for i, pos in enumerate(self.open_positions):
            # Update max/min prices
            if pos.direction == 1:
                pos.max_price = max(pos.max_price, row['high'])
                pos.min_price = min(pos.min_price, row['low'])
            else:
                pos.max_price = max(pos.max_price, row['high'])
                pos.min_price = min(pos.min_price, row['low'])
            
            # Check stop loss
            if pos.direction == 1 and row['low'] <= pos.stop_loss:
                positions_to_close.append((i, pos.stop_loss, "stop_loss"))
            elif pos.direction == -1 and row['high'] >= pos.stop_loss:
                positions_to_close.append((i, pos.stop_loss, "stop_loss"))
            
            # Check take profit
            elif pos.direction == 1 and row['high'] >= pos.take_profit:
                positions_to_close.append((i, pos.take_profit, "take_profit"))
            elif pos.direction == -1 and row['low'] <= pos.take_profit:
                positions_to_close.append((i, pos.take_profit, "take_profit"))
            
            # Check trailing stop
            elif self.trailing_stop and pos.trailing_stop > 0:
                if pos.direction == 1 and row['low'] <= pos.trailing_stop:
                    positions_to_close.append((i, pos.trailing_stop, "trailing_stop"))
                elif pos.direction == -1 and row['high'] >= pos.trailing_stop:
                    positions_to_close.append((i, pos.trailing_stop, "trailing_stop"))
            
            # Update trailing stop
            if self.trailing_stop:
                self._update_trailing_stop(pos, row)
            
            # Check max holding time
            holding_minutes = (timestamp - pos.entry_time).total_seconds() / 60
            if holding_minutes >= self.max_holding_minutes:
                positions_to_close.append((i, row['close'], "max_holding"))
        
        # Close positions (reverse order to maintain indices)
        for i, exit_price, reason in reversed(positions_to_close):
            self._close_position(i, exit_price, timestamp, reason)
    
    def _update_trailing_stop(self, pos: Position, row: pd.Series) -> None:
        """Update trailing stop."""
        if pos.direction == 1:
            # Long position
            profit_pct = (row['high'] - pos.entry_price) / pos.entry_price
            if profit_pct >= self.trailing_activation:
                new_trailing = row['high'] * (1 - self.trailing_distance)
                if pos.trailing_stop == 0 or new_trailing > pos.trailing_stop:
                    pos.trailing_stop = new_trailing
        else:
            # Short position
            profit_pct = (pos.entry_price - row['low']) / pos.entry_price
            if profit_pct >= self.trailing_activation:
                new_trailing = row['low'] * (1 + self.trailing_distance)
                if pos.trailing_stop == 0 or new_trailing < pos.trailing_stop:
                    pos.trailing_stop = new_trailing
    
    def _close_position(
        self,
        index: int,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
    ) -> None:
        """Close a position."""
        pos = self.open_positions.pop(index)
        
        # Apply slippage on exit
        exit_price = exit_price * (1 - self.slippage * pos.direction)
        
        # Calculate P&L
        if pos.direction == 1:
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
        
        # Commission
        commission = (pos.entry_price + exit_price) * pos.size * self.commission
        pnl -= commission
        
        pnl_pct = pnl / (pos.entry_price * pos.size)
        
        holding_minutes = (exit_time - pos.entry_time).total_seconds() / 60
        
        # Max favorable/adverse
        if pos.direction == 1:
            max_fav = (pos.max_price - pos.entry_price) / pos.entry_price
            max_adv = (pos.entry_price - pos.min_price) / pos.entry_price
        else:
            max_fav = (pos.entry_price - pos.min_price) / pos.entry_price
            max_adv = (pos.max_price - pos.entry_price) / pos.entry_price
        
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            direction=pos.direction,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=self.slippage,
            max_favorable=max_fav,
            max_adverse=max_adv,
            holding_minutes=int(holding_minutes),
            exit_reason=reason,
        )
        
        self.trades.append(trade)
        self.capital += pnl
        self.daily_pnl += pnl
        
        logger.debug(f"Closed {pos.direction} at {exit_price:.5f}, P&L: {pnl:.2f} ({pnl_pct:.4%})")
    
    def _close_all_positions(self, row: pd.Series, timestamp: pd.Timestamp, reason: str) -> None:
        """Close all open positions."""
        while self.open_positions:
            self._close_position(0, row['close'], timestamp, reason)
    
    def _record_equity(self, timestamp: pd.Timestamp, row: pd.Series) -> None:
        """Record current equity."""
        # Calculate unrealized P&L
        unrealized = 0
        for pos in self.open_positions:
            if pos.direction == 1:
                unrealized += (row['close'] - pos.entry_price) * pos.size
            else:
                unrealized += (pos.entry_price - row['close']) * pos.size
        
        equity = self.capital + unrealized
        self.equity_curve.append(equity)
    
    def _is_allowed_session(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is in allowed trading session."""
        hour = timestamp.hour
        minute = timestamp.minute
        time_minutes = hour * 60 + minute
        
        sessions = {
            'london': (8 * 60, 17 * 60),
            'new_york': (13 * 60, 22 * 60),
            'overlap': (13 * 60, 17 * 60),
            'asian': (0, 8 * 60),
        }
        
        for session in self.allowed_sessions:
            if session in sessions:
                start, end = sessions[session]
                if start <= time_minutes < end:
                    return True
        
        return False
    
    def _calculate_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backtest results."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'holding_minutes': t.holding_minutes,
            'exit_reason': t.exit_reason,
            'max_favorable': t.max_favorable,
            'max_adverse': t.max_adverse,
        } for t in self.trades])
        
        equity_series = pd.Series(self.equity_curve, index=df.index[:len(self.equity_curve)])
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        n_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else np.inf
        
        # Drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Returns
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(1440 * 252) if returns.std() > 0 else 0
        sortino = returns.mean() / (returns[returns < 0].std() + 1e-10) * np.sqrt(1440 * 252) if len(returns[returns < 0]) > 0 else 0
        
        # Consecutive wins/losses
        trades_df['win'] = trades_df['pnl'] > 0
        max_consec_wins = self._max_consecutive(trades_df['win'], True)
        max_consec_losses = self._max_consecutive(trades_df['win'], False)
        
        # Risk metrics
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': trades_df['pnl'].mean(),
            'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses,
            'avg_holding_minutes': trades_df['holding_minutes'].mean(),
            'var_95': var_95,
            'cvar_95': cvar_95,
            'trades': trades_df,
            'equity_curve': equity_series,
            'daily_stats': self.daily_stats,
        }
    
    def _max_consecutive(self, series: pd.Series, value: bool) -> int:
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
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Get detailed trade analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct * 100,
            'commission': t.commission,
            'holding_minutes': t.holding_minutes,
            'exit_reason': t.exit_reason,
            'max_favorable_pct': t.max_favorable * 100,
            'max_adverse_pct': t.max_adverse * 100,
        } for t in self.trades])


if __name__ == "__main__":
    import yfinance as yf
    
    # Test backtest
    df = yf.download('GC=F', period='1mo', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    # Generate random signals for testing
    np.random.seed(42)
    signals = pd.Series(np.random.choice([-1, 0, 1], size=len(df), p=[0.1, 0.8, 0.1]), index=df.index)
    confidence = pd.Series(np.random.uniform(0.5, 1.0, len(df)), index=df.index)
    
    engine = BacktestEngine()
    results = engine.run(df, signals, confidence)
    
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
