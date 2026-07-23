"""Streamlit dashboard for XAUUSD Scalping System."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..config import get_config
from ..data import DataLoader
from ..features import FeatureEngineer
from ..models.persistence import ModelPersistence
from ..signals.generator import SignalGenerator, SignalType
from ..backtesting.engine import BacktestEngine
from ..evaluation.metrics import calculate_metrics

st.set_page_config(
    page_title="XAUUSD AI Scalping System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=60)
def load_data(interval: str = "1m", period: str = "1d"):
    """Load and cache market data."""
    loader = DataLoader()
    return loader.load_ohlcv(interval=interval, period=period)


@st.cache_data(ttl=300)
def load_model(model_name: str = "ensemble"):
    """Load and cache model."""
    persistence = ModelPersistence()
    try:
        return persistence.load_latest(model_name)
    except Exception:
        return None


@st.cache_data(ttl=60)
def generate_features(df: pd.DataFrame):
    """Generate and cache features."""
    engineer = FeatureEngineer()
    return engineer.transform(df)


@st.cache_data(ttl=60)
def generate_signals(features: pd.DataFrame, ohlcv: pd.DataFrame, model_data: dict):
    """Generate and cache signals."""
    if model_data is None:
        return [], pd.Series(), pd.Series()
    
    generator = SignalGenerator(
        model=model_data['model'],
        scaler=model_data.get('scaler'),
        feature_names=model_data.get('feature_names'),
    )
    
    signals = generator.generate(features, ohlcv)
    signal_series = generator.get_signal_series(signals)
    confidence_series = generator.get_confidence_series(signals)
    
    return signals, signal_series, confidence_series


def create_price_chart(df: pd.DataFrame, signals: pd.Series = None):
    """Create candlestick chart with signals."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price', 'Volume', 'Signals'),
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='XAUUSD',
        ),
        row=1, col=1,
    )
    
    if signals is not None:
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, 'low'] * 0.9995,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='BUY',
                ),
                row=1, col=1,
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, 'high'] * 1.0005,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='SELL',
                ),
                row=1, col=1,
            )
    
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'),
        row=2, col=1,
    )
    
    if signals is not None:
        fig.add_trace(
            go.Bar(
                x=signals.index,
                y=signals.values,
                marker_color=['green' if v == 1 else 'red' if v == -1 else 'gray' for v in signals.values],
                name='Signal',
            ),
            row=3, col=1,
        )
    
    fig.update_layout(
        title='XAUUSD Price & Signals',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark',
    )
    
    return fig


def create_equity_chart(equity: pd.Series, drawdown: pd.Series = None):
    """Create equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown'),
    )
    
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            line=dict(color='#00ff88', width=1),
            name='Equity',
        ),
        row=1, col=1,
    )
    
    if drawdown is not None:
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)',
                name='Drawdown %',
            ),
            row=2, col=1,
        )
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True,
    )
    
    return fig


def create_metrics_dashboard(metrics: dict):
    """Create metrics display."""
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
    
    with cols[1]:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
        st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
    
    with cols[2]:
        st.metric("Total Trades", f"{metrics.get('n_trades', 0)}")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        st.metric("Expectancy", f"{metrics.get('expectancy', 0):.4f}")
        st.metric("Risk/Reward", f"{metrics.get('risk_reward_ratio', 0):.2f}")
    
    with cols[3]:
        st.metric("Avg Win", f"{metrics.get('avg_win', 0):.2f}")
        st.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}")
        st.metric("Max Consec Wins", f"{metrics.get('max_consecutive_wins', 0)}")
        st.metric("Max Consec Losses", f"{metrics.get('max_consecutive_losses', 0)}")


def create_trade_analysis(trades: pd.DataFrame):
    """Create trade analysis charts."""
    if trades.empty:
        st.info("No trades to analyze")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades['pnl'],
            nbinsx=30,
            name='PnL Distribution',
            marker_color='blue',
            opacity=0.7,
        ))
        fig.update_layout(title='PnL Distribution', template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'exit_reason' in trades.columns:
            fig = go.Figure()
            for reason in trades['exit_reason'].unique():
                subset = trades[trades['exit_reason'] == reason]
                fig.add_trace(go.Box(
                    y=subset['pnl'],
                    name=reason,
                    boxpoints='outliers',
                ))
            fig.update_layout(title='PnL by Exit Reason', template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    trades_sorted = trades.sort_values('entry_time')
    trades_sorted['cum_pnl'] = trades_sorted['pnl'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_sorted['entry_time'],
        y=trades_sorted['cum_pnl'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#00ff88', width=2),
    ))
    fig.update_layout(title='Cumulative PnL', template='plotly_dark', height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    st.markdown("""
        <style>
        .stMetric > div > div > div { color: #00ff88 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 XAUUSD AI Scalping Signal System")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=0)
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo"], index=1)
        model_name = st.selectbox("Model", ["ensemble", "xgboost", "lightgbm", "catboost", "random_forest"], index=0)
        
        st.markdown("---")
        st.header("📊 Signal Filters")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.55, 0.01)
        min_risk_reward = st.slider("Min Risk/Reward", 1.0, 3.0, 1.5, 0.1)
        
        st.markdown("---")
        auto_refresh = st.checkbox("Auto Refresh (60s)", value=True)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading market data..."):
        df = load_data(interval=interval, period=period)
    
    if df.empty:
        st.error("No market data available")
        return
    
    # Load model
    model_data = load_model(model_name)
    
    if not model_data:
        st.warning(f"Model '{model_name}' not found. Please train a model first.")
        st.info("Run `python scripts/train.py` to train models.")
        return
    
    # Generate features
    with st.spinner("Generating features..."):
        features = generate_features(df)
    
    if features.empty:
        st.error("Feature generation failed.")
        return
    
    # Generate signals
    with st.spinner("Generating signals..."):
        signals, signal_series, confidence_series = generate_signals(features, df, model_data)
    
    # Current signal status
    st.header("🎯 Current Signal")
    
    if len(signals) > 0:
        latest_signal = signals[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        signal_icons = {
            SignalType.BUY: "🟢 BUY",
            SignalType.SELL: "🔴 SELL",
            SignalType.WAIT: "⚪ WAIT",
        }
        
        with col1:
            st.metric("Signal", signal_icons.get(latest_signal.signal_type, "UNKNOWN"))
        
        with col2:
            st.metric("Confidence", f"{latest_signal.confidence:.2%}")
        
        with col3:
            st.metric("Price", f"${latest_signal.price:.2f}")
        
        with col4:
            st.metric("Time", latest_signal.timestamp.strftime("%H:%M:%S"))
    
    # Price chart
    st.header("📊 Price Chart")
    price_chart = create_price_chart(df.tail(500), signal_series.tail(500) if signal_series is not None else None)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Performance metrics from backtest
    if signal_series is not None and len(signal_series) > 0:
        st.header("📈 Performance Metrics")
        
        with st.spinner("Running backtest..."):
            engine = BacktestEngine()
            results = engine.run(df, signal_series, confidence_series)
            
            if 'error' not in results:
                metrics = {
                    'total_return': results.get('total_return', 0),
                    'annualized_return': results.get('total_return', 0) * 252,
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'sortino_ratio': results.get('sortino_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'calmar_ratio': results.get('total_return', 0) / abs(results.get('max_drawdown', 1)) if results.get('max_drawdown', 0) != 0 else 0,
                    'volatility': df['close'].pct_change().std() * np.sqrt(1440 * 252),
                    'win_rate': results.get('win_rate', 0),
                    'n_trades': results.get('n_trades', 0),
                    'profit_factor': results.get('profit_factor', 0),
                    'expectancy': results.get('expectancy', 0),
                    'risk_reward_ratio': abs(results.get('avg_win', 1) / results.get('avg_loss', 1)) if results.get('avg_loss', 0) != 0 else 0,
                    'avg_win': results.get('avg_win', 0),
                    'avg_loss': results.get('avg_loss', 0),
                    'max_consecutive_wins': results.get('max_consecutive_wins', 0),
                    'max_consecutive_losses': results.get('max_consecutive_losses', 0),
                }
                
                create_metrics_dashboard(metrics)
                
                # Equity curve
                if 'equity_curve' in results:
                    st.header("📉 Equity Curve")
                    equity = results['equity_curve']
                    drawdown = (equity / equity.expanding().max() - 1) * 100
                    equity_chart = create_equity_chart(equity, drawdown)
                    st.plotly_chart(equity_chart, use_container_width=True)
                
                # Trade analysis
                if 'trades' in results and not results['trades'].empty:
                    st.header("🔍 Trade Analysis")
                    create_trade_analysis(results['trades'])
            else:
                st.warning("No trades executed in backtest period.")
    else:
        st.info("No signals generated. Waiting for market conditions...")
    
    # Signal history
    st.header("📜 Recent Signals")
    if len(signals) > 0:
        signal_data = []
        for s in signals[-50:]:
            if s.signal_type != SignalType.WAIT:
                signal_data.append({
                    'Time': s.timestamp.strftime("%Y-%m-%d %H:%M"),
                    'Signal': s.signal_type.name,
                    'Confidence': f"{s.confidence:.2%}",
                    'Price': f"${s.price:.2f}",
                })
        
        if signal_data:
            signal_df = pd.DataFrame(signal_data)
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active signals in recent history")
    
    # Model info
    with st.expander("🤖 Model Information"):
        if model_data:
            metadata = model_data.get('metadata', {})
            st.json({
                'model_name': metadata.get('model_name', 'Unknown'),
                'version': metadata.get('version', 'Unknown'),
                'timestamp': metadata.get('timestamp', 'Unknown'),
                'n_features': metadata.get('n_features', 'Unknown'),
                'metrics': metadata.get('metrics', {}),
            })
        else:
            st.warning("No model loaded")
    
    # Data quality
    with st.expander("📊 Data Quality"):
        loader = DataLoader()
        quality = loader.validate_data_quality(df)
        st.json(quality)
    
    # Feature importance
    with st.expander("🎯 Feature Importance"):
        if model_data and 'feature_importance' in model_data:
            fi = model_data['feature_importance']
            if isinstance(fi, pd.DataFrame) and not fi.empty:
                st.bar_chart(fi.set_index('feature')['importance'].head(20))
            else:
                st.info("Feature importance not available")
        else:
            st.info("Feature importance not available")


if __name__ == "__main__":
    main()
