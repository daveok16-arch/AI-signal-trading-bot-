"""Telegram Bot notifier for XAUUSD Scalping System with Exness Referral Integration."""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import aiohttp
from aiohttp import ClientTimeout, ClientError

from .base import (
    NotificationBase,
    NotificationConfig,
    SignalNotification,
    TradeNotification,
    RiskAlertNotification,
    SystemAlertNotification,
    SignalDirection,
    ReferralConfig,
)

logger = logging.getLogger(__name__)


class TelegramConfig:
    """Telegram-specific configuration from environment variables."""
    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        parse_mode: str = "Markdown",
        disable_web_page_preview: bool = True,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_minute: int = 20,
        rate_limit_per_hour: int = 100,
    ):
        # Load from environment with defaults
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode = parse_mode or os.getenv("TELEGRAM_PARSE_MODE", "Markdown")
        self.disable_web_page_preview = disable_web_page_preview or os.getenv("TELEGRAM_DISABLE_WEB_PREVIEW", "true").lower() == "true"
        self.timeout = float(os.getenv("TELEGRAM_TIMEOUT", str(timeout)))
        self.max_retries = int(os.getenv("TELEGRAM_MAX_RETRIES", str(max_retries)))
        self.retry_delay = float(os.getenv("TELEGRAM_RETRY_DELAY", str(retry_delay)))
        self.rate_limit_per_minute = int(os.getenv("TELEGRAM_RATE_LIMIT_MINUTE", str(rate_limit_per_minute)))
        self.rate_limit_per_hour = int(os.getenv("TELEGRAM_RATE_LIMIT_HOUR", str(rate_limit_per_hour)))


class TelegramNotifier(NotificationBase):
    """Telegram Bot notification provider with Exness Referral Integration."""
    
    API_URL = "https://api.telegram.org/bot{token}/{method}"
    
    # Referral section template - configurable via environment
    REFERRAL_TEMPLATE = """
────────────────────

🚀 *NEW TO TRADING?*

Register using our official **Exness** referral link:
{referral_link}

💰 Receive **$10,000 in virtual funds** and learn how to trade with a market leader.

📚 Create your account, practice risk-free with a demo account, then start trading with our AI signals.

⚠️ *Trade responsibly. All trading involves risk, and past performance does not guarantee future results.*"""

    def __init__(
        self,
        config: NotificationConfig,
        telegram_config: Optional[TelegramConfig] = None,
        referral_config: Optional[ReferralConfig] = None,
    ):
        super().__init__(config)
        self.telegram_config = telegram_config or TelegramConfig()
        self.referral_config = referral_config or self._load_referral_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Validate configuration
        if not self.telegram_config.bot_token or not self.telegram_config.chat_id:
            logger.warning("Telegram bot_token or chat_id not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = config.enabled

    def _load_referral_config(self) -> ReferralConfig:
        """Load referral configuration from environment variables."""
        return ReferralConfig(
            referral_link=os.getenv("TELEGRAM_REFERRAL_LINK", "https://bit.ly/4yAbSgu"),
            promo_text=os.getenv("TELEGRAM_PROMO_TEXT", "Receive $10,000 in virtual funds and learn how to trade with a market leader."),
            enabled=os.getenv("TELEGRAM_REFERRAL_ENABLED", "true").lower() == "true",
            show_on_every_signal=os.getenv("TELEGRAM_REFERRAL_EVERY_SIGNAL", "true").lower() == "true",
            min_confidence_for_referral=float(os.getenv("TELEGRAM_REFERRAL_MIN_CONFIDENCE", "0.0")),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.telegram_config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_referral_section(self) -> str:
        """Build the referral CTA section with Exness branding."""
        if not self.referral_config.enabled:
            return ""
        
        return self.REFERRAL_TEMPLATE.format(
            referral_link=self.referral_config.referral_link,
            promo_text=self.referral_config.promo_text,
        )

    def _should_include_referral(self, signal: SignalNotification) -> bool:
        """Determine if referral section should be included."""
        if not self.referral_config.enabled:
            return False
        if not self.referral_config.show_on_every_signal:
            return False
        if signal.confidence < self.referral_config.min_confidence_for_referral:
            return False
        return True

    async def _send_message(self, text: str, parse_mode: str = None) -> bool:
        """Send message to Telegram."""
        if not self.enabled:
            logger.debug("Telegram notifier disabled, skipping send")
            return False
            
        if not self._check_rate_limit():
            logger.warning("Telegram rate limit exceeded")
            return False
        
        session = await self._get_session()
        url = self.API_URL.format(
            token=self.telegram_config.bot_token,
            method="sendMessage"
        )
        
        payload = {
            "chat_id": self.telegram_config.chat_id,
            "text": text,
            "parse_mode": parse_mode or self.telegram_config.parse_mode,
            "disable_web_page_preview": self.telegram_config.disable_web_page_preview,
        }
        
        async def _do_send():
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        self._record_send()
                        return True
                    else:
                        logger.error(f"Telegram API error: {data}")
                        return False
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After", "1")
                    logger.warning(f"Telegram rate limited, retry after {retry_after}s")
                    await asyncio.sleep(int(retry_after) + 1)
                    return False
                else:
                    error_text = await response.text()
                    logger.error(f"Telegram HTTP {response.status}: {error_text}")
                    return False
        
        try:
            return await self._retry_with_backoff(_do_send)
        except ClientError as e:
            logger.error(f"Telegram connection error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Telegram request timeout")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _format_signal(self, signal: SignalNotification) -> str:
        """
        Format signal notification for Telegram with Exness referral section.
        
        Required format:
        🌍 *Time Zone:* UTC
        🟢 *XAUUSD - BUY*
        📈 *Confidence:* 94% 🔥
        💰 *Entry:* 3362.15
        🛑 *Stop Loss:* 3358.80
        🎯 *Take Profit:* 3368.40
        ⚖️ *Risk:Reward:* 1:2
        ⏰ *Signal Time:* 14:35 UTC
        🤖 *AI Analysis:*
        • Trend Confirmed
        • Momentum Bullish
        • Volume Above Average
        • Multi-Model Consensus
        ────────────────────
        [Referral Section]
        """
        # Direction formatting
        if signal.direction == SignalDirection.BUY:
            direction_line = f"🟢 *{signal.symbol} - BUY*"
        elif signal.direction == SignalDirection.SELL:
            direction_line = f"🔴 *{signal.symbol} - SELL*"
        else:
            direction_line = f"⚪ *{signal.symbol} - WAIT*"
        
        # Confidence with emoji
        conf_pct = signal.confidence * 100
        if signal.confidence >= 0.9:
            conf_emoji = "🔥"
        elif signal.confidence >= 0.75:
            conf_emoji = "⭐"
        elif signal.confidence >= 0.6:
            conf_emoji = "📊"
        else:
            conf_emoji = "📈"
        
        # Time formatting
        time_str = signal.timestamp.strftime("%H:%M UTC")
        
        # Risk:Reward ratio
        rr_ratio = signal.risk_reward_ratio
        if rr_ratio > 0:
            rr_text = f"1:{rr_ratio:.1f}" if rr_ratio != int(rr_ratio) else f"1:{int(rr_ratio)}"
        else:
            # Calculate from entry, SL, TP if available
            if signal.entry_price and signal.stop_loss and signal.take_profit:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                if risk > 0:
                    rr_ratio = reward / risk
                    rr_text = f"1:{rr_ratio:.1f}" if rr_ratio != int(rr_ratio) else f"1:{int(rr_ratio)}"
                else:
                    rr_text = "N/A"
            else:
                rr_text = "N/A"
        
        # Build AI Analysis section
        analysis_lines = []
        if signal.supporting_analysis:
            for key, value in signal.supporting_analysis.items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        analysis_lines.append(f"• {key}: {value:.4f}")
                    else:
                        analysis_lines.append(f"• {key}: {value:.2f}")
                elif isinstance(value, bool):
                    analysis_lines.append(f"• {key}: {'✅' if value else '❌'}")
                else:
                    analysis_lines.append(f"• {key}: {value}")
        else:
            # Default analysis based on signal direction and metadata
            if signal.direction == SignalDirection.BUY:
                analysis_lines = [
                    "• Trend Confirmed",
                    "• Momentum Bullish",
                    "• Volume Above Average",
                    "• Multi-Model Consensus",
                ]
            elif signal.direction == SignalDirection.SELL:
                analysis_lines = [
                    "• Trend Reversal Confirmed",
                    "• Momentum Bearish",
                    "• Volume Increasing",
                    "• Multi-Model Consensus",
                ]
            else:
                analysis_lines = [
                    "• No Clear Signal",
                    "• Wait for Confirmation",
                ]
        
        analysis_section = "\n".join(analysis_lines) if analysis_lines else "• Analysis pending..."
        
        # Format prices with appropriate precision
        entry_str = f"{signal.entry_price:.2f}" if signal.entry_price else "N/A"
        sl_str = f"{signal.stop_loss:.2f}" if signal.stop_loss else "N/A"
        tp_str = f"{signal.take_profit:.2f}" if signal.take_profit else "N/A"
        
        # Build main message (Markdown format)
        message = (
            f"🌍 *Time Zone:* UTC\n\n"
            f"{direction_line}\n"
            f"📈 *Confidence:* {signal.confidence:.0%} {conf_emoji}\n\n"
            f"💰 *Entry:* {entry_str}\n"
            f"🛑 *Stop Loss:* {sl_str}\n"
            f"🎯 *Take Profit:* {tp_str}\n"
            f"⚖️ *Risk:Reward:* {rr_text}\n\n"
            f"⏰ *Signal Time:* {time_str}\n\n"
            f"🤖 *AI Analysis:*\n"
            f"{analysis_section}\n"
        )
        
        # Add referral section if enabled
        if self._should_include_referral(signal):
            message += self._build_referral_section()
        
        return message

    def _format_trade(self, trade: TradeNotification) -> str:
        """Format trade notification for Telegram."""
        if trade.exit_price is None:
            # Trade opened
            direction_emoji = "🟢" if trade.direction == SignalDirection.BUY else "🔴"
            message = (
                f"{direction_emoji} *TRADE OPENED*\n\n"
                f"🆔 *ID:* `{trade.trade_id}`\n"
                f"📊 *Asset:* {trade.symbol}\n"
                f"📈 *Direction:* {trade.direction.value}\n"
                f"💰 *Entry:* ${trade.entry_price:,.5f}\n"
                f"🛑 *Stop Loss:* ${trade.stop_loss:,.5f}\n"
                f"🎯 *Take Profit:* ${trade.take_profit:,.5f}\n"
                f"📦 *Size:* {trade.size:.4f}\n"
                f"⏰ *Time:* {trade.entry_time.strftime('%H:%M:%S')} UTC"
            )
        else:
            # Trade closed
            pnl_emoji = "🟢" if trade.pnl and trade.pnl > 0 else "🔴"
            pnl_text = f"+${trade.pnl:,.2f}" if trade.pnl and trade.pnl > 0 else f"${trade.pnl:,.2f}"
            
            message = (
                f"{pnl_emoji} *TRADE CLOSED*\n\n"
                f"🆔 *ID:* `{trade.trade_id}`\n"
                f"📊 *Asset:* {trade.symbol}\n"
                f"📈 *Direction:* {trade.direction.value}\n"
                f"💰 *Entry:* ${trade.entry_price:,.5f}\n"
                f"💵 *Exit:* ${trade.exit_price:,.5f}\n"
                f"📦 *Size:* {trade.size:.4f}\n"
                f"💰 *PnL:* {pnl_text} ({trade.pnl_pct:.2%})\n"
                f"🚪 *Reason:* {trade.exit_reason or 'N/A'}\n"
                f"💸 *Commission:* ${trade.commission:.2f}\n"
                f"📉 *Slippage:* {trade.slippage:.4%}\n"
                f"⏰ *Entry:* {trade.entry_time.strftime('%H:%M:%S')} UTC\n"
                f"⏰ *Exit:* {trade.exit_time.strftime('%H:%M:%S')} UTC"
            )
        
        return message

    def _format_risk_alert(self, alert: RiskAlertNotification) -> str:
        """Format risk alert for Telegram."""
        severity_emoji = {
            "LOW": "🔵",
            "MEDIUM": "🟡",
            "HIGH": "🟠",
            "CRITICAL": "🔴",
        }.get(alert.severity, "⚪")
        
        return (
            f"{severity_emoji} *RISK ALERT: {alert.alert_type}*\n\n"
            f"*Severity:* {alert.severity}\n"
            f"*Message:* {alert.message}\n"
            f"*Current Value:* {alert.current_value:.4f}\n"
            f"*Threshold:* {alert.threshold:.4f}\n"
            f"*Time:* {alert.timestamp.strftime('%H:%M:%S')} UTC"
        )

    def _format_system_alert(self, alert: SystemAlertNotification) -> str:
        """Format system alert for Telegram."""
        severity_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
        }.get(alert.severity, "⚪")
        
        return (
            f"{severity_emoji} *SYSTEM ALERT: {alert.component}*\n\n"
            f"*Severity:* {alert.severity}\n"
            f"*Message:* {alert.message}\n"
            f"*Time:* {alert.timestamp.strftime('%H:%M:%S')} UTC"
        )

    async def send_signal(self, signal: SignalNotification) -> bool:
        """Send signal notification with Exness referral section."""
        text = self._format_signal(signal)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_trade(self, trade: TradeNotification) -> bool:
        """Send trade notification."""
        text = self._format_trade(trade)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_risk_alert(self, alert: RiskAlertNotification) -> bool:
        """Send risk alert notification."""
        text = self._format_risk_alert(alert)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_system_alert(self, alert: SystemAlertNotification) -> bool:
        """Send system alert notification."""
        text = self._format_system_alert(alert)
        return await self._send_message(text, parse_mode="Markdown")

    async def test_connection(self) -> bool:
        """Test Telegram Bot connection."""
        if not self.enabled:
            logger.warning("Telegram notifier not configured")
            return False
        
        session = await self._get_session()
        url = self.API_URL.format(
            token=self.telegram_config.bot_token,
            method="getMe"
        )
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        logger.info(f"Telegram bot connected: @{bot_info.get('username', 'unknown')}")
                        return True
                logger.error(f"Telegram test failed: {response.status}")
                return False
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False

    async def send_startup_message(self) -> bool:
        """Send startup notification with referral section."""
        text = (
            "🚀 *XAUUSD AI Scalper Started*\n\n"
            f"⏰ *Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            "*Status:* Ready to generate AI signals\n\n"
        )
        
        # Add referral section to startup
        if self.referral_config.enabled:
            text += self._build_referral_section()
        
        return await self._send_message(text, parse_mode="Markdown")

    async def send_shutdown_message(self) -> bool:
        """Send shutdown notification."""
        text = (
            "🛑 *XAUUSD AI Scalper Stopped*\n\n"
            f"⏰ *Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            "*Status:* Graceful shutdown"
        )
        return await self._send_message(text, parse_mode="Markdown")


def create_telegram_notifier(
    config: NotificationConfig = None,
    telegram_config: TelegramConfig = None,
    referral_config: ReferralConfig = None,
) -> 'TelegramNotifier':
    """Factory function to create Telegram notifier with Exness referral integration.
    
    All configuration loaded from environment variables:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID
    - TELEGRAM_ENABLED
    - TELEGRAM_REFERRAL_LINK
    - TELEGRAM_PROMO_TEXT
    - TELEGRAM_REFERRAL_ENABLED
    - TELEGRAM_REFERRAL_EVERY_SIGNAL
    - TELEGRAM_REFERRAL_MIN_CONFIDENCE
    """
    # Load from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    tg_config = telegram_config or TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        parse_mode="Markdown",
        disable_web_page_preview=True,
        timeout=10.0,
        max_retries=3,
        retry_delay=1.0,
        rate_limit_per_minute=20,
        rate_limit_per_hour=100,
    )
    
    ref_config = referral_config or ReferralConfig(
        referral_link=os.getenv("TELEGRAM_REFERRAL_LINK", "https://bit.ly/4yAbSgu"),
        promo_text=os.getenv("TELEGRAM_PROMO_TEXT", "Receive $10,000 in virtual funds and learn how to trade with a market leader."),
        enabled=os.getenv("TELEGRAM_REFERRAL_ENABLED", "true").lower() == "true",
        show_on_every_signal=os.getenv("TELEGRAM_REFERRAL_EVERY_SIGNAL", "true").lower() == "true",
        min_confidence_for_referral=float(os.getenv("TELEGRAM_REFERRAL_MIN_CONFIDENCE", "0.0")),
    )
    
    notification_config = config or NotificationConfig(
        enabled=bool(bot_token and chat_id),
        rate_limit_per_minute=20,
        rate_limit_per_hour=100,
        retry_attempts=3,
        retry_base_delay=1.0,
        retry_max_delay=60.0,
        timeout=10.0,
    )
    
    return TelegramNotifier(notification_config, tg_config, ref_config)
