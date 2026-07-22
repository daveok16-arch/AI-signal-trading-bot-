"""Base notification classes for XAUUSD Scalping System."""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    SIGNAL = "signal"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    RISK_ALERT = "risk_alert"
    SYSTEM_ALERT = "system_alert"
    MODEL_UPDATE = "model_update"
    ERROR = "error"


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


@dataclass
class ReferralConfig:
    """Referral program configuration - loaded from environment variables."""
    referral_link: str = "https://bit.ly/4yAbSgu"
    promo_text: str = "Receive $10,000 in virtual funds and learn how to trade with a market leader."
    enabled: bool = True
    show_on_every_signal: bool = True
    min_confidence_for_referral: float = 0.0


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled: bool = True
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 200
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    timeout: float = 10.0
    parse_mode: str = "Markdown"
    disable_web_page_preview: bool = True


@dataclass
class SignalNotification:
    """Signal notification data."""
    symbol: str
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    timeframe: str = "1m"
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 0.0
    supporting_analysis: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.risk_reward_ratio == 0.0 and self.stop_loss != self.entry_price:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                self.risk_reward_ratio = round(reward / risk, 2)


@dataclass
class TradeNotification:
    """Trade execution notification."""
    trade_id: str
    symbol: str
    direction: SignalDirection
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_reason: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class RiskAlertNotification:
    """Risk management alert."""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemAlertNotification:
    """System-level alert."""
    component: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationBase(ABC):
    """Abstract base class for notification providers."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self._rate_limit_minute: List[datetime] = []
        self._rate_limit_hour: List[datetime] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def send_signal(self, notification: SignalNotification) -> bool:
        """Send signal notification."""
        pass

    @abstractmethod
    async def send_trade(self, notification: TradeNotification) -> bool:
        """Send trade notification."""
        pass

    @abstractmethod
    async def send_risk_alert(self, notification: RiskAlertNotification) -> bool:
        """Send risk alert."""
        pass

    @abstractmethod
    async def send_system_alert(self, notification: SystemAlertNotification) -> bool:
        """Send system alert."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test notification channel connectivity."""
        pass

    def _check_rate_limit(self) -> bool:
        """Check if rate limits allow sending."""
        now = datetime.utcnow()
        
        minute_ago = now.timestamp() - 60
        hour_ago = now.timestamp() - 3600
        
        self._rate_limit_minute = [t for t in self._rate_limit_minute if t.timestamp() > minute_ago]
        self._rate_limit_hour = [t for t in self._rate_limit_hour if t.timestamp() > hour_ago]
        
        if len(self._rate_limit_minute) >= self.config.rate_limit_per_minute:
            self.logger.warning("Rate limit exceeded: per minute")
            return False
        
        if len(self._rate_limit_hour) >= self.config.rate_limit_per_hour:
            self.logger.warning("Rate limit exceeded: per hour")
            return False
        
        return True

    def _record_send(self):
        """Record a send for rate limiting."""
        now = datetime.utcnow()
        self._rate_limit_minute.append(now)
        self._rate_limit_hour.append(now)

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    delay = min(
                        self.config.retry_base_delay * (2 ** attempt),
                        self.config.retry_max_delay
                    )
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts failed: {e}")
        
        raise last_exception


class NotificationManager:
    """Manages multiple notification providers."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.providers: List[NotificationBase] = []
        self.logger = logging.getLogger(f"{__name__}.NotificationManager")

    def add_provider(self, provider: NotificationBase):
        """Add a notification provider."""
        self.providers.append(provider)
        self.logger.info(f"Added notification provider: {provider.__class__.__name__}")

    async def send_signal(self, notification: SignalNotification) -> Dict[str, bool]:
        """Send signal to all providers."""
        results = {}
        for provider in self.providers:
            try:
                result = await provider.send_signal(notification)
                results[provider.__class__.__name__] = result
            except Exception as e:
                self.logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                results[provider.__class__.__name__] = False
        return results

    async def send_trade(self, notification: TradeNotification) -> Dict[str, bool]:
        """Send trade notification to all providers."""
        results = {}
        for provider in self.providers:
            try:
                result = await provider.send_trade(notification)
                results[provider.__class__.__name__] = result
            except Exception as e:
                self.logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                results[provider.__class__.__name__] = False
        return results

    async def send_risk_alert(self, notification: RiskAlertNotification) -> Dict[str, bool]:
        """Send risk alert to all providers."""
        results = {}
        for provider in self.providers:
            try:
                result = await provider.send_risk_alert(notification)
                results[provider.__class__.__name__] = result
            except Exception as e:
                self.logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                results[provider.__class__.__name__] = False
        return results

    async def send_system_alert(self, notification: SystemAlertNotification) -> Dict[str, bool]:
        """Send system alert to all providers."""
        results = {}
        for provider in self.providers:
            try:
                result = await provider.send_system_alert(notification)
                results[provider.__class__.__name__] = result
            except Exception as e:
                self.logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                results[provider.__class__.__name__] = False
        return results

    async def test_all(self) -> Dict[str, bool]:
        """Test all providers."""
        results = {}
        for provider in self.providers:
            try:
                results[provider.__class__.__name__] = await provider.test_connection()
            except Exception as e:
                self.logger.error(f"Provider {provider.__class__.__name__} test failed: {e}")
                results[provider.__class__.__name__] = False
        return results
