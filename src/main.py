"""Main entry point for XAUUSD AI Scalping System."""
import asyncio
import argparse
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import load_config_from_env, AppSettings
from src.core.lifecycle import Application, LifecycleManager, ApplicationState
from src.notifications.telegram import create_telegram_notifier
from src.notifications.base import NotificationManager, NotificationConfig
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.persistence import ModelPersistence
from src.signals.generator import SignalGenerator, SignalType
from src.backtesting.engine import BacktestEngine
from src.observability import setup_logging, get_logger

logger = get_logger(__name__)


class ScalperApp:
    """Main application class."""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.app = Application("XAUUSD Scalper", settings)
        self.notification_manager: Optional[NotificationManager] = None
        self.telegram_notifier = None
        self.data_loader = None
        self.feature_engineer = None
        self.signal_generator = None
        self.current_signals = []
        self._running = False
        self._signal_task = None
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing XAUUSD Scalper...")
        
        # Setup notifications
        await self._setup_notifications()
        
        # Initialize data components
        self.data_loader = DataLoader(self.settings.config_path)
        self.feature_engineer = FeatureEngineer(self.settings.config_path)
        
        # Load model
        await self._load_model()
        
        # Register lifecycle hooks
        self.app.add_startup("notifications", self._startup_notifications, priority=100)
        self.app.add_startup("data_loader", self._startup_data_loader, priority=90)
        self.app.add_startup("model", self._startup_model, priority=80)
        self.app.add_startup("signal_generator", self._startup_signal_generator, priority=70)
        
        self.app.add_shutdown("notifications", self._shutdown_notifications, priority=100)
        self.app.add_shutdown("signal_task", self._shutdown_signal_task, priority=90)
        self.app.add_shutdown("data_loader", self._shutdown_data_loader, priority=80)
        self.app.add_shutdown("model", self._shutdown_model, priority=70)
        
        self.app.add_health_check("data", self._health_check_data, priority=100)
        self.app.add_health_check("model", self._health_check_model, priority=90)
        self.app.add_health_check("telegram", self._health_check_telegram, priority=80)
        self.app.add_health_check("disk", self._health_check_disk, priority=70)
        
        logger.info("Initialization complete")
    
    async def _setup_notifications(self):
        """Setup notification providers."""
        self.notification_manager = NotificationManager(NotificationConfig(
            enabled=self.settings.telegram.enabled,
            rate_limit_per_minute=self.settings.telegram.rate_limit_per_minute,
            rate_limit_per_hour=self.settings.telegram.rate_limit_per_hour,
            retry_attempts=self.settings.telegram.max_retries,
            retry_base_delay=self.settings.telegram.retry_delay,
            retry_max_delay=60.0,
            timeout=self.settings.telegram.timeout,
        ))
        
        if self.settings.telegram.enabled and self.settings.telegram.is_configured():
            self.telegram_notifier = create_telegram_notifier(
                NotificationConfig(
                    enabled=True,
                    rate_limit_per_minute=self.settings.telegram.rate_limit_per_minute,
                    rate_limit_per_hour=self.settings.telegram.rate_limit_per_hour,
                    retry_attempts=self.settings.telegram.max_retries,
                    retry_base_delay=self.settings.telegram.retry_delay,
                    retry_max_delay=60.0,
                    timeout=self.settings.telegram.timeout,
                )
            )
            self.notification_manager.add_provider(self.telegram_notifier)
            
            # Test connection
            connected = await self.telegram_notifier.test_connection()
            if connected:
                await self.telegram_notifier.send_startup_message()
                logger.info("Telegram notifications enabled and tested")
            else:
                logger.warning("Telegram connection test failed")
        else:
            logger.info("Telegram notifications disabled or not configured")
    
    async def _load_model(self):
        """Load the trained model."""
        try:
            persistence = ModelPersistence(self.settings.model.model_dir)
            loaded = persistence.load_latest(self.settings.model.default_model)
            
            if loaded:
                self.signal_generator = SignalGenerator(
                    model=loaded['model'],
                    config_path=self.settings.config_path,
                    scaler=loaded.get('scaler'),
                    feature_names=loaded.get('feature_names'),
                )
                logger.info(f"Loaded model: {self.settings.model.default_model}")
            else:
                logger.warning("No model found. Train a model first.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    # Lifecycle hooks
    async def _startup_notifications(self):
        """Send startup notifications."""
        if self.telegram_notifier:
            await self.telegram_notifier.send_startup_message()
    
    async def _startup_data_loader(self):
        """Initialize data loader."""
        # Test data connection
        df = self.data_loader.load_ohlcv(
            interval=self.settings.data.interval,
            period=self.settings.data.period,
            use_cache=True,
        )
        if df.empty:
            raise RuntimeError("Failed to load initial market data")
        logger.info(f"Data loader ready: {len(df)} bars loaded")
    
    async def _startup_model(self):
        """Verify model is loaded."""
        if self.signal_generator is None:
            logger.warning("No model loaded - signals will not be generated")
        else:
            logger.info("Model ready for signal generation")
    
    async def _startup_signal_generator(self):
        """Start signal generation task."""
        if self.signal_generator and self.settings.signal_check_interval > 0:
            self._signal_task = asyncio.create_task(self._signal_generation_loop())
            logger.info(f"Signal generation started (interval: {self.settings.signal_check_interval}s)")
    
    async def _shutdown_notifications(self):
        """Send shutdown notifications."""
        if self.telegram_notifier:
            await self.telegram_notifier.send_shutdown_message()
            await self.telegram_notifier.close()
    
    async def _shutdown_signal_task(self):
        """Stop signal generation task."""
        if self._signal_task:
            self._signal_task.cancel()
            try:
                await self._signal_task
            except asyncio.CancelledError:
                pass
            logger.info("Signal generation task stopped")
    
    async def _shutdown_data_loader(self):
        """Cleanup data loader."""
        logger.info("Data loader shutdown")
    
    async def _shutdown_model(self):
        """Cleanup model."""
        logger.info("Model shutdown")
    
    # Health checks
    async def _health_check_data(self) -> dict:
        """Check data source health."""
        try:
            df = self.data_loader.load_ohlcv(
                interval=self.settings.data.interval,
                period="1d",
                use_cache=True,
            )
            return {"status": "healthy", "bars": len(df), "last_bar": str(df.index[-1]) if not df.empty else "none"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _health_check_model(self) -> dict:
        """Check model health."""
        if self.signal_generator is None:
            return {"status": "unhealthy", "error": "No model loaded"}
        return {"status": "healthy", "model": self.settings.model.default_model}
    
    async def _health_check_telegram(self) -> dict:
        """Check Telegram connection."""
        if not self.telegram_notifier:
            return {"status": "disabled"}
        connected = await self.telegram_notifier.test_connection()
        return {"status": "healthy" if connected else "unhealthy", "connected": connected}
    
    async def _health_check_disk(self) -> dict:
        """Check disk space."""
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_pct = (free / total) * 100
        return {
            "status": "healthy" if free_pct > 10 else "degraded" if free_pct > 5 else "unhealthy",
            "free_gb": round(free / (1024**3), 2),
            "used_pct": round((used / total) * 100, 1),
        }
    
    async def _signal_generation_loop(self):
        """Main signal generation loop."""
        logger.info("Starting signal generation loop")
        
        while True:
            try:
                await asyncio.sleep(self.settings.signal_check_interval)
                
                if not self.signal_generator:
                    logger.debug("No model loaded, skipping signal generation")
                    continue
                
                # Load fresh data
                df = self.data_loader.load_ohlcv(
                    interval=self.settings.data.interval,
                    period="1d",
                    use_cache=True,
                )
                
                if df.empty:
                    logger.warning("No data available for signal generation")
                    continue
                
                # Generate features
                features = self.feature_engineer.transform(df)
                
                if features.empty:
                    logger.warning("No features generated")
                    continue
                
                # Generate signals
                signals = self.signal_generator.generate(features, df)
                
                # Process new signals
                for signal in signals:
                    if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                        await self._handle_new_signal(signal, df)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in signal generation loop: {e}")
                # Send error alert
                if self.telegram_notifier:
                    from notifications.base import SystemAlertNotification
                    alert = SystemAlertNotification(
                        component="signal_generation",
                        severity="ERROR",
                        message=f"Signal generation error: {str(e)}",
                        timestamp=datetime.utcnow(),
                    )
                    await self.notification_manager.send_system_alert(alert)
                
                await asyncio.sleep(60)  # Back off on error
    
    async def _handle_new_signal(self, signal, df):
        """Handle a new trading signal."""
        logger.info(f"New {signal.signal_type.value} signal: {signal.confidence:.1%} confidence")
        
        # Prepare notification
        from notifications.base import SignalNotification, SignalDirection
        
        # Calculate stop loss and take profit
        entry = signal.price
        if signal.signal_type == SignalType.BUY:
            stop_loss = entry * (1 - self.settings.backtest.stop_loss_pct)
            take_profit = entry * (1 + self.settings.backtest.take_profit_pct)
        else:
            stop_loss = entry * (1 + self.settings.backtest.stop_loss_pct)
            take_profit = entry * (1 - self.settings.backtest.take_profit_pct)
        
        notification = SignalNotification(
            symbol=self.settings.data.symbol,
            direction=SignalDirection.BUY if signal.signal_type == SignalType.BUY else SignalDirection.SELL,
            confidence=signal.confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=signal.timestamp,
            timeframe=self.settings.data.interval,
            risk_reward_ratio=abs(take_profit - entry) / abs(entry - stop_loss) if entry != stop_loss else 0,
            position_size_pct=self.settings.backtest.position_size,
            supporting_analysis={
                "trend": "bullish" if signal.signal_type == SignalType.BUY else "bearish",
                "volume_ratio": signal.metadata.get("volume_ratio", 0),
                "atr": signal.metadata.get("atr", 0),
                "rsi": signal.metadata.get("rsi", 50),
            },
            metadata={
                "model": self.settings.model.default_model,
                "features_count": len(signal.metadata.get("features", [])),
            },
        )
        
        # Send notification
        results = await self.notification_manager.send_signal(notification)
        logger.info(f"Notification sent: {results}")
        
        # Store signal
        self.current_signals.append(signal)
        # Keep only last 100 signals
        if len(self.current_signals) > 100:
            self.current_signals = self.current_signals[-100:]
    
    async def run(self):
        """Run the application."""
        async with self.app:
            logger.info("Application running. Press Ctrl+C to stop.")
            try:
                # Keep running
                while self.app.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
            finally:
                await self.app.stop("Keyboard interrupt")


async def run_worker(settings: AppSettings):
    """Run as background worker for signal generation."""
    logger.info("Starting XAUUSD Scalper Worker...")
    
    app = ScalperApp(settings)
    await app.initialize()
    
    # Run signal generation once for worker mode
    if settings.backtest_on_startup:
        await run_backtest(settings)
    else:
        # Continuous signal generation
        await app.run()


async def run_api(settings: AppSettings):
    """Run API server."""
    logger.info("Starting API server...")
    import uvicorn
    from api.main import app as fastapi_app
    
    config = uvicorn.Config(
        fastapi_app,
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        timeout_keep_alive=settings.api.timeout,
        log_level=settings.logging.level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_backtest(settings: AppSettings):
    """Run backtest on startup."""
    logger.info("Running backtest on startup...")
    
    # Load model
    persistence = ModelPersistence(settings.model.model_dir)
    loaded = persistence.load_latest(settings.model.default_model)
    
    if not loaded:
        logger.error("No model found for backtest")
        return
    
    # Load data
    data_loader = DataLoader(settings.config_path)
    df = data_loader.load_ohlcv(
        interval=settings.data.interval,
        period=settings.data.period,
    )
    
    if df.empty:
        logger.error("No data for backtest")
        return
    
    # Generate features
    engineer = FeatureEngineer(settings.config_path)
    features = engineer.transform(df)
    
    # Generate signals
    signal_generator = SignalGenerator(
        model=loaded['model'],
        config_path=settings.config_path,
        scaler=loaded.get('scaler'),
        feature_names=loaded.get('feature_names'),
    )
    signals = signal_generator.generate(features, df)
    signal_series = signal_generator.get_signal_series(signals)
    confidence_series = signal_generator.get_confidence_series(signals)
    
    # Run backtest
    engine = BacktestEngine(settings.config_path)
    results = engine.run(df, signal_series, confidence_series)
    
    logger.info(f"Backtest complete: {results.get('n_trades', 0)} trades, "
                f"return: {results.get('total_return', 0):.2%}, "
                f"sharpe: {results.get('sharpe_ratio', 0):.2f}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="XAUUSD AI Scalping System")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--worker", action="store_true", help="Run as background worker")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--dashboard", action="store_true", help="Run dashboard")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--json-log", action="store_true", help="JSON log format")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        json_format=args.json_log,
    )
    
    # Load configuration
    settings = load_config_from_env(args.config)
    
    # Override from CLI
    if args.log_level:
        settings.logging.level = args.log_level
    if args.json_log:
        settings.logging.json_format = True
    
    # Run appropriate mode
    if args.worker:
        await run_worker(settings)
    elif args.api:
        await run_api(settings)
    elif args.backtest:
        await run_backtest(settings)
    elif args.dashboard:
        # Dashboard runs via streamlit directly
        logger.info("Run dashboard with: streamlit run src/dashboard/app.py")
    elif args.train:
        logger.info("Run training with: python scripts/train.py")
    else:
        # Default: run full application
        app = ScalperApp(settings)
        await app.initialize()
        await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
