"""Core configuration for XAUUSD Scalping System."""
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with type conversion."""
    value = os.environ.get(key, default)
    
    if value is None and required:
        raise ValueError(f"Required environment variable {key} not set")
    
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    elif isinstance(default, int):
        return int(value) if value is not None else default
    elif isinstance(default, float):
        return float(value) if value is not None else default
    elif isinstance(default, list):
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        return value or default
    elif isinstance(default, dict):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {key} as JSON, using default")
                return default
        return value or default
    
    return value


@dataclass
class TelegramSettings:
    """Telegram Bot settings from environment."""
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 20
    rate_limit_per_hour: int = 100
    
    def __post_init__(self):
        # Load from environment
        self.enabled = get_env('TELEGRAM_ENABLED', self.enabled)
        self.bot_token = get_env('TELEGRAM_BOT_TOKEN', self.bot_token)
        self.chat_id = get_env('TELEGRAM_CHAT_ID', self.chat_id)
        self.parse_mode = get_env('TELEGRAM_PARSE_MODE', self.parse_mode)
        self.disable_web_page_preview = get_env('TELEGRAM_DISABLE_WEB_PREVIEW', self.disable_web_page_preview)
        self.timeout = get_env('TELEGRAM_TIMEOUT', self.timeout)
        self.max_retries = get_env('TELEGRAM_MAX_RETRIES', self.max_retries)
        self.retry_delay = get_env('TELEGRAM_RETRY_DELAY', self.retry_delay)
        self.rate_limit_per_minute = get_env('TELEGRAM_RATE_LIMIT_MINUTE', self.rate_limit_per_minute)
        self.rate_limit_per_hour = get_env('TELEGRAM_RATE_LIMIT_HOUR', self.rate_limit_per_hour)
    
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)


@dataclass
class APISettings:
    """API server settings from environment."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    auth_enabled: bool = False
    api_key: str = ""
    rate_limit: int = 100
    
    def __post_init__(self):
        self.host = get_env('API_HOST', self.host)
        self.port = get_env('API_PORT', self.port)
        self.workers = get_env('API_WORKERS', self.workers)
        self.timeout = get_env('API_TIMEOUT', self.timeout)
        self.cors_origins = get_env('API_CORS_ORIGINS', self.cors_origins)
        self.auth_enabled = get_env('API_AUTH_ENABLED', self.auth_enabled)
        self.api_key = get_env('API_KEY', self.api_key)
        self.rate_limit = get_env('API_RATE_LIMIT', self.rate_limit)


@dataclass
class DashboardSettings:
    """Dashboard settings from environment."""
    host: str = "0.0.0.0"
    port: int = 8501
    theme: str = "dark"
    refresh_interval: int = 30
    auto_refresh: bool = True
    
    def __post_init__(self):
        self.host = get_env('DASHBOARD_HOST', self.host)
        self.port = get_env('DASHBOARD_PORT', self.port)
        self.theme = get_env('DASHBOARD_THEME', self.theme)
        self.refresh_interval = get_env('DASHBOARD_REFRESH_INTERVAL', self.refresh_interval)
        self.auto_refresh = get_env('DASHBOARD_AUTO_REFRESH', self.auto_refresh)


@dataclass
class LoggingSettings:
    """Logging settings from environment."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = False
    file: str = "logs/xauusd_scalper.log"
    max_bytes: int = 10485760
    backup_count: int = 10
    console: bool = True
    
    def __post_init__(self):
        self.level = get_env('LOG_LEVEL', self.level)
        self.format = get_env('LOG_FORMAT', self.format)
        self.json_format = get_env('LOG_JSON_FORMAT', self.json_format)
        self.file = get_env('LOG_FILE', self.file)
        self.max_bytes = get_env('LOG_MAX_BYTES', self.max_bytes)
        self.backup_count = get_env('LOG_BACKUP_COUNT', self.backup_count)
        self.console = get_env('LOG_CONSOLE', self.console)


@dataclass
class DataSettings:
    """Data acquisition settings from environment."""
    symbol: str = "GC=F"
    interval: str = "1m"
    period: str = "7d"
    cache_dir: str = "data/cache"
    cache_ttl_minutes: int = 5
    max_retries: int = 3
    retry_delay: float = 5.0
    timezone: str = "UTC"
    
    def __post_init__(self):
        self.symbol = get_env('DATA_SYMBOL', self.symbol)
        self.interval = get_env('DATA_INTERVAL', self.interval)
        self.period = get_env('DATA_PERIOD', self.period)
        self.cache_dir = get_env('DATA_CACHE_DIR', self.cache_dir)
        self.cache_ttl_minutes = get_env('DATA_CACHE_TTL', self.cache_ttl_minutes)
        self.max_retries = get_env('DATA_MAX_RETRIES', self.max_retries)
        self.retry_delay = get_env('DATA_RETRY_DELAY', self.retry_delay)
        self.timezone = get_env('DATA_TIMEZONE', self.timezone)


@dataclass
class BacktestSettings:
    """Backtesting settings from environment."""
    initial_capital: float = 100000.0
    position_size: float = 0.1
    max_positions: int = 1
    commission: float = 0.0001
    slippage: float = 0.0001
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.01
    trailing_stop: bool = True
    trailing_activation: float = 0.005
    trailing_distance: float = 0.003
    max_holding_minutes: int = 15
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    risk_per_trade: float = 0.01
    position_sizing: str = "kelly"
    kelly_fraction: float = 0.5
    min_confidence: float = 0.55
    min_risk_reward: float = 1.5
    max_spread_pct: float = 0.0005
    
    def __post_init__(self):
        self.initial_capital = get_env('BACKTEST_INITIAL_CAPITAL', self.initial_capital)
        self.position_size = get_env('BACKTEST_POSITION_SIZE', self.position_size)
        self.max_positions = get_env('BACKTEST_MAX_POSITIONS', self.max_positions)
        self.commission = get_env('BACKTEST_COMMISSION', self.commission)
        self.slippage = get_env('BACKTEST_SLIPPAGE', self.slippage)
        self.stop_loss_pct = get_env('BACKTEST_STOP_LOSS', self.stop_loss_pct)
        self.take_profit_pct = get_env('BACKTEST_TAKE_PROFIT', self.take_profit_pct)
        self.trailing_stop = get_env('BACKTEST_TRAILING_STOP', self.trailing_stop)
        self.trailing_activation = get_env('BACKTEST_TRAILING_ACTIVATION', self.trailing_activation)
        self.trailing_distance = get_env('BACKTEST_TRAILING_DISTANCE', self.trailing_distance)
        self.max_holding_minutes = get_env('BACKTEST_MAX_HOLDING', self.max_holding_minutes)
        self.max_daily_trades = get_env('BACKTEST_MAX_DAILY_TRADES', self.max_daily_trades)
        self.max_daily_loss_pct = get_env('BACKTEST_MAX_DAILY_LOSS', self.max_daily_loss_pct)
        self.max_drawdown_pct = get_env('BACKTEST_MAX_DRAWDOWN', self.max_drawdown_pct)
        self.risk_per_trade = get_env('BACKTEST_RISK_PER_TRADE', self.risk_per_trade)
        self.position_sizing = get_env('BACKTEST_POSITION_SIZING', self.position_sizing)
        self.kelly_fraction = get_env('BACKTEST_KELLY_FRACTION', self.kelly_fraction)
        self.min_confidence = get_env('BACKTEST_MIN_CONFIDENCE', self.min_confidence)
        self.min_risk_reward = get_env('BACKTEST_MIN_RISK_REWARD', self.min_risk_reward)
        self.max_spread_pct = get_env('BACKTEST_MAX_SPREAD', self.max_spread_pct)


@dataclass
class SignalSettings:
    """Signal generation settings from environment."""
    confidence_threshold: float = 0.55
    min_risk_reward: float = 1.5
    consensus_threshold: float = 0.6
    voting_method: str = "weighted"
    signal_smoothing: int = 3
    min_hold_minutes: int = 2
    max_signals_per_hour: int = 12
    cooldown_minutes: int = 5
    trend_filter_enabled: bool = True
    trend_ma_period: int = 200
    trend_slope_threshold: float = 0.0001
    volatility_filter_enabled: bool = True
    atr_threshold: float = 0.001
    volume_filter_enabled: bool = True
    volume_ma_period: int = 20
    volume_threshold: float = 1.2
    session_filter_enabled: bool = True
    allowed_sessions: List[str] = field(default_factory=lambda: ["london", "new_york", "overlap"])
    news_filter_enabled: bool = False
    news_api_key: str = ""
    news_impact_threshold: str = "high"
    
    def __post_init__(self):
        self.confidence_threshold = get_env('SIGNAL_CONFIDENCE_THRESHOLD', self.confidence_threshold)
        self.min_risk_reward = get_env('SIGNAL_MIN_RISK_REWARD', self.min_risk_reward)
        self.consensus_threshold = get_env('SIGNAL_CONSENSUS_THRESHOLD', self.consensus_threshold)
        self.voting_method = get_env('SIGNAL_VOTING_METHOD', self.voting_method)
        self.signal_smoothing = get_env('SIGNAL_SMOOTHING', self.signal_smoothing)
        self.min_hold_minutes = get_env('SIGNAL_MIN_HOLD', self.min_hold_minutes)
        self.max_signals_per_hour = get_env('SIGNAL_MAX_PER_HOUR', self.max_signals_per_hour)
        self.cooldown_minutes = get_env('SIGNAL_COOLDOWN', self.cooldown_minutes)
        self.trend_filter_enabled = get_env('SIGNAL_TREND_FILTER', self.trend_filter_enabled)
        self.trend_ma_period = get_env('SIGNAL_TREND_MA', self.trend_ma_period)
        self.trend_slope_threshold = get_env('SIGNAL_TREND_SLOPE', self.trend_slope_threshold)
        self.volatility_filter_enabled = get_env('SIGNAL_VOL_FILTER', self.volatility_filter_enabled)
        self.atr_threshold = get_env('SIGNAL_ATR_THRESHOLD', self.atr_threshold)
        self.volume_filter_enabled = get_env('SIGNAL_VOLUME_FILTER', self.volume_filter_enabled)
        self.volume_ma_period = get_env('SIGNAL_VOLUME_MA', self.volume_ma_period)
        self.volume_threshold = get_env('SIGNAL_VOLUME_THRESHOLD', self.volume_threshold)
        self.session_filter_enabled = get_env('SIGNAL_SESSION_FILTER', self.session_filter_enabled)
        self.allowed_sessions = get_env('SIGNAL_ALLOWED_SESSIONS', self.allowed_sessions)
        self.news_filter_enabled = get_env('SIGNAL_NEWS_FILTER', self.news_filter_enabled)
        self.news_api_key = get_env('SIGNAL_NEWS_API_KEY', self.news_api_key)
        self.news_impact_threshold = get_env('SIGNAL_NEWS_IMPACT', self.news_impact_threshold)


@dataclass
class ModelSettings:
    """Model settings from environment."""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    class_weight: str = "balanced"
    cv_folds: int = 5
    cv_strategy: str = "time_series_split"
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.0001
    feature_importance_top_n: int = 20
    save_feature_importance: bool = True
    save_predictions: bool = True
    save_models: bool = True
    model_dir: str = "models"
    
    def __post_init__(self):
        self.test_size = get_env('MODEL_TEST_SIZE', self.test_size)
        self.validation_size = get_env('MODEL_VAL_SIZE', self.validation_size)
        self.random_state = get_env('MODEL_RANDOM_STATE', self.random_state)
        self.stratify = get_env('MODEL_STRATIFY', self.stratify)
        self.class_weight = get_env('MODEL_CLASS_WEIGHT', self.class_weight)
        self.cv_folds = get_env('MODEL_CV_FOLDS', self.cv_folds)
        self.cv_strategy = get_env('MODEL_CV_STRATEGY', self.cv_strategy)
        self.early_stopping = get_env('MODEL_EARLY_STOPPING', self.early_stopping)
        self.patience = get_env('MODEL_PATIENCE', self.patience)
        self.min_delta = get_env('MODEL_MIN_DELTA', self.min_delta)
        self.feature_importance_top_n = get_env('MODEL_FEATURE_TOP_N', self.feature_importance_top_n)
        self.save_feature_importance = get_env('MODEL_SAVE_FEATURE_IMPORTANCE', self.save_feature_importance)
        self.save_predictions = get_env('MODEL_SAVE_PREDICTIONS', self.save_predictions)
        self.save_models = get_env('MODEL_SAVE_MODELS', self.save_models)
        self.model_dir = get_env('MODEL_DIR', self.model_dir)


@dataclass
class WalkForwardSettings:
    """Walk-forward validation settings from environment."""
    enabled: bool = True
    train_window: int = 5000
    test_window: int = 1000
    step_size: int = 500
    min_train_size: int = 3000
    expanding_window: bool = False
    purge_gap: int = 10
    embargo_pct: float = 0.01
    
    def __post_init__(self):
        self.enabled = get_env('WF_ENABLED', self.enabled)
        self.train_window = get_env('WF_TRAIN_WINDOW', self.train_window)
        self.test_window = get_env('WF_TEST_WINDOW', self.test_window)
        self.step_size = get_env('WF_STEP_SIZE', self.step_size)
        self.min_train_size = get_env('WF_MIN_TRAIN', self.min_train_size)
        self.expanding_window = get_env('WF_EXPANDING', self.expanding_window)
        self.purge_gap = get_env('WF_PURGE_GAP', self.purge_gap)
        self.embargo_pct = get_env('WF_EMBARGO', self.embargo_pct)


@dataclass
class TargetSettings:
    """Target/labeling settings from environment."""
    type: str = "classification"
    horizon: int = 5
    threshold_long: float = 0.0015
    threshold_short: float = -0.0015
    neutral_zone: float = 0.0005
    labeling_method: str = "triple_barrier"
    triple_barrier_upper_mult: float = 2.0
    triple_barrier_lower_mult: float = 2.0
    triple_barrier_max_holding: int = 15
    triple_barrier_vol_lookback: int = 20
    
    def __post_init__(self):
        self.type = get_env('TARGET_TYPE', self.type)
        self.horizon = get_env('TARGET_HORIZON', self.horizon)
        self.threshold_long = get_env('TARGET_THRESHOLD_LONG', self.threshold_long)
        self.threshold_short = get_env('TARGET_THRESHOLD_SHORT', self.threshold_short)
        self.neutral_zone = get_env('TARGET_NEUTRAL_ZONE', self.neutral_zone)
        self.labeling_method = get_env('TARGET_LABELING_METHOD', self.labeling_method)
        self.triple_barrier_upper_mult = get_env('TARGET_TB_UPPER', self.triple_barrier_upper_mult)
        self.triple_barrier_lower_mult = get_env('TARGET_TB_LOWER', self.triple_barrier_lower_mult)
        self.triple_barrier_max_holding = get_env('TARGET_TB_MAX_HOLD', self.triple_barrier_max_holding)
        self.triple_barrier_vol_lookback = get_env('TARGET_TB_VOL_LOOKBACK', self.triple_barrier_vol_lookback)


@dataclass
class AppSettings:
    """Main application settings aggregating all sub-configurations."""
    # Environment
    environment: str = "production"
    debug: bool = False
    
    # Sub-configurations
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    api: APISettings = field(default_factory=APISettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    signals: SignalSettings = field(default_factory=SignalSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    walk_forward: WalkForwardSettings = field(default_factory=WalkForwardSettings)
    target: TargetSettings = field(default_factory=TargetSettings)
    
    def __post_init__(self):
        self.environment = get_env('APP_ENV', self.environment)
        self.debug = get_env('APP_DEBUG', self.debug)
    
    def validate(self) -> List[str]:
        """Validate settings and return list of warnings/errors."""
        issues = []
        
        if self.telegram.enabled and not self.telegram.is_configured():
            issues.append("Telegram enabled but bot_token or chat_id not configured")
        
        if self.api.auth_enabled and not self.api.api_key:
            issues.append("API auth enabled but no API_KEY set")
        
        if self.dashboard.port == self.api.port:
            issues.append(f"Dashboard and API ports conflict: both on {self.api.port}")
        
        if self.data.cache_ttl_minutes < 1:
            issues.append("Data cache TTL should be at least 1 minute")
        
        if self.backtest.initial_capital <= 0:
            issues.append("Backtest initial capital must be positive")
        
        if self.signals.confidence_threshold < 0 or self.signals.confidence_threshold > 1:
            issues.append("Signal confidence threshold must be between 0 and 1")
        
        return issues


def load_config_from_env(config_path: Optional[str] = None) -> AppSettings:
    """Load application configuration from environment variables and optional config file."""
    settings = AppSettings()
    
    # If config file provided, load it (for backward compatibility)
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            # Note: Environment variables take precedence over file config
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
    
    # Validate and warn
    issues = settings.validate()
    for issue in issues:
        logger.warning(f"Config validation: {issue}")
    
    return settings
