"""Logging configuration for XAUUSD Scalping System."""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from ..config import get_config


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)


class TradingLogger:
    """Specialized logger for trading events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_signal(self, signal: Dict[str, Any]) -> None:
        """Log trading signal."""
        self.logger.info(
            "SIGNAL_GENERATED",
            extra={
                'event_type': 'signal',
                'signal_type': signal.get('type'),
                'symbol': signal.get('symbol', 'XAUUSD'),
                'price': signal.get('price'),
                'confidence': signal.get('confidence'),
                'timestamp': signal.get('timestamp'),
            }
        )
    
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log trade execution."""
        self.logger.info(
            "TRADE_EXECUTED",
            extra={
                'event_type': 'trade',
                'trade_id': trade.get('id'),
                'direction': trade.get('direction'),
                'entry_price': trade.get('entry_price'),
                'exit_price': trade.get('exit_price'),
                'size': trade.get('size'),
                'pnl': trade.get('pnl'),
                'commission': trade.get('commission'),
                'duration_minutes': trade.get('duration_minutes'),
                'exit_reason': trade.get('exit_reason'),
            }
        )
    
    def log_position_update(self, position: Dict[str, Any]) -> None:
        """Log position update."""
        self.logger.debug(
            "POSITION_UPDATE",
            extra={
                'event_type': 'position_update',
                'symbol': position.get('symbol', 'XAUUSD'),
                'direction': position.get('direction'),
                'entry_price': position.get('entry_price'),
                'current_price': position.get('current_price'),
                'unrealized_pnl': position.get('unrealized_pnl'),
                'stop_loss': position.get('stop_loss'),
                'take_profit': position.get('take_profit'),
            }
        )
    
    def log_risk_event(self, event: Dict[str, Any]) -> None:
        """Log risk management event."""
        self.logger.warning(
            "RISK_EVENT",
            extra={
                'event_type': 'risk',
                'risk_type': event.get('type'),
                'message': event.get('message'),
                'current_value': event.get('current_value'),
                'limit': event.get('limit'),
            }
        )
    
    def log_model_event(self, event: Dict[str, Any]) -> None:
        """Log model training/prediction event."""
        self.logger.info(
            "MODEL_EVENT",
            extra={
                'event_type': 'model',
                'model_name': event.get('model_name'),
                'event_subtype': event.get('subtype'),
                'metrics': event.get('metrics'),
                'data_shape': event.get('data_shape'),
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with context."""
        extra = {'event_type': 'error', 'error_type': type(error).__name__}
        if context:
            extra.update(context)
        
        self.logger.error(
            f"ERROR: {error}",
            exc_info=True,
            extra=extra,
        )


def setup_logging(
    config_path: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    json_format: bool = False,
) -> None:
    """Setup application logging."""
    config = get_config(config_path) if config_path else None
    
    if config:
        log_config = config.get_section('logging')
        log_level = log_config.get('level', log_level)
        log_file = log_file or log_config.get('file', 'logs/xauusd_scalper.log')
        json_format = log_config.get('json_format', json_format)
        max_bytes = log_config.get('max_bytes', 10485760)
        backup_count = log_config.get('backup_count', 10)
        console = log_config.get('console', True)
    else:
        max_bytes = 10485760
        backup_count = 10
        console = True
    
    # Create log directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels if configured
    if config:
        log_levels = log_config.get('log_levels', {})
        for logger_name, level in log_levels.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))
    
    # Log startup
    root_logger.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def get_trading_logger(name: str) -> TradingLogger:
    """Get trading logger instance."""
    return TradingLogger(get_logger(name))


# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance."""
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.debug(
                    f"{func.__name__} completed in {duration:.3f}s",
                    extra={'function': func.__name__, 'duration': duration}
                )
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s: {e}",
                    exc_info=True,
                    extra={'function': func.__name__, 'duration': duration, 'error': str(e)}
                )
                raise
        return wrapper
    return decorator


# Context manager for logging
class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, *args):
        logging.setLogRecordFactory(self.old_factory)


if __name__ == "__main__":
    # Test logging
    setup_logging(log_level="DEBUG", json_format=True)
    logger = get_logger("test")
    trading_logger = get_trading_logger("trading")
    
    logger.info("Test message")
    logger.warning("Test warning")
    
    trading_logger.log_signal({
        'type': 'BUY',
        'price': 2000.50,
        'confidence': 0.75,
        'timestamp': '2024-01-01T12:00:00Z',
    })
    
    trading_logger.log_trade({
        'id': 'trade_123',
        'direction': 'LONG',
        'entry_price': 2000.00,
        'exit_price': 2005.00,
        'size': 1.0,
        'pnl': 5.00,
        'commission': 0.20,
        'duration_minutes': 10,
        'exit_reason': 'take_profit',
    })
