#!/usr/bin/env python3
"""Backtesting script for XAUUSD Scalping System."""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data import DataLoader
from features import FeatureEngineer
from models.persistence import ModelPersistence
from signals.generator import SignalGenerator
from backtesting.engine import BacktestEngine
from evaluation.metrics import generate_performance_summary
from logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Scalping Backtest")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--model", type=str, default="ensemble", help="Model name")
    parser.add_argument("--interval", type=str, default="1m", help="Data interval")
    parser.add_argument("--period", type=str, default="7d", help="Data period")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    
    args = parser.parse_args()
    
    setup_logging(config_path=args.config, log_level="INFO")
    logger = get_logger("backtest")
    
    logger.info("Starting XAUUSD Scalping Backtest")
    
    # Load data
    logger.info("Loading market data...")
    loader = DataLoader(args.config)
    
    if args.start and args.end:
        df = loader.load_ohlcv(interval=args.interval, start=args.start, end=args.end)
    else:
        df = loader.load_ohlcv(interval=args.interval, period=args.period)
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(df)} bars")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    persistence = ModelPersistence()
    
    try:
        loaded = persistence.load_latest(args.model)
        model = loaded['model']
        scaler = loaded.get('scaler')
        feature_names = loaded.get('feature_names')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer(args.config)
    features = engineer.transform(df)
    
    if feature_names:
        features = features[feature_names]
    
    if features.empty:
        logger.error("Feature generation failed.")
        return 1
    
    # Generate signals
    logger.info("Generating signals...")
    generator = SignalGenerator(model=model, scaler=scaler, feature_names=feature_names)
    signals = generator.generate(features, df)
    
    signal_series = generator.get_signal_series(signals)
    confidence_series = generator.get_confidence_series(signals)
    
    logger.info(f"Signal distribution: {signal_series.value_counts().to_dict()}")
    
    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(args.config)
    results = engine.run(df, signal_series, confidence_series)
    
    if 'error' in results:
        logger.error(f"Backtest failed: {results['error']}")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(generate_performance_summary(results))
    
    # Trade analysis
    if 'trades' in results and not results['trades'].empty:
        trades = results['trades']
        print(f"\nTotal Trades: {len(trades)}")
        print(f"Winning: {len(trades[trades['pnl'] > 0])}")
        print(f"Losing: {len(trades[trades['pnl'] < 0])}")
        print(f"Avg Holding: {trades['holding_minutes'].mean():.1f} minutes")
        
        if 'exit_reason' in trades.columns:
            print("\nExit Reasons:")
            for reason, count in trades['exit_reason'].value_counts().items():
                print(f"  {reason}: {count}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                serializable[k] = v.to_dict('records') if isinstance(v, pd.DataFrame) else v.to_dict()
            elif isinstance(v, (np.integer, np.floating)):
                serializable[k] = float(v)
            elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                serializable[k] = str(v)
            else:
                serializable[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    # Generate report
    if args.report:
        report_path = Path("reports") / f"backtest_{args.model}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Backtest Report\n\n")
            f.write(f"**Model:** {args.model}\n")
            f.write(f"**Period:** {df.index[0]} to {df.index[-1]}\n")
            f.write(f"**Interval:** {args.interval}\n\n")
            f.write("## Performance Summary\n\n")
            f.write(generate_performance_summary(results).replace('\n', '\n\n'))
            
            if 'trades' in results and not results['trades'].empty:
                f.write("\n\n## Trade Analysis\n\n")
                trades = results['trades']
                f.write(f"- Total Trades: {len(trades)}\n")
                f.write(f"- Winning: {len(trades[trades['pnl'] > 0])}\n")
                f.write(f"- Losing: {len(trades[trades['pnl'] < 0])}\n")
                f.write(f"- Avg PnL: {trades['pnl'].mean():.2f}\n")
                f.write(f"- Max Win: {trades['pnl'].max():.2f}\n")
                f.write(f"- Max Loss: {trades['pnl'].min():.2f}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    # Generate plots
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plot_dir = Path("plots")
            plot_dir.mkdir(exist_ok=True)
            
            if 'equity_curve' in results:
                equity = results['equity_curve']
                plt.figure(figsize=(12, 6))
                plt.plot(equity.index, equity.values)
                plt.title('Equity Curve')
                plt.xlabel('Time')
                plt.ylabel('Equity')
                plt.grid(True)
                plt.savefig(plot_dir / 'equity_curve.png', dpi=150)
                plt.close()
                
                drawdown = (equity / equity.expanding().max() - 1) * 100
                plt.figure(figsize=(12, 4))
                plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
                plt.title('Drawdown')
                plt.xlabel('Time')
                plt.ylabel('Drawdown %')
                plt.grid(True)
                plt.savefig(plot_dir / 'drawdown.png', dpi=150)
                plt.close()
            
            if 'trades' in results and not results['trades'].empty:
                trades = results['trades']
                plt.figure(figsize=(10, 6))
                plt.hist(trades['pnl'], bins=30, edgecolor='black', alpha=0.7)
                plt.title('PnL Distribution')
                plt.xlabel('PnL')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(plot_dir / 'pnl_distribution.png', dpi=150)
                plt.close()
            
            logger.info(f"Plots saved to {plot_dir}")
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
    
    logger.info("Backtest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
