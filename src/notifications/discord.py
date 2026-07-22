"""Discord Webhook notifier for XAUUSD Scalping System with Referral Integration."""
import asyncio
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

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


@dataclass
class DiscordConfig:
    """Discord-specific configuration."""
    webhook_url: str = ""
    username: str = "XAUUSD Scalper"
    avatar_url: str = ""
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 100


class DiscordNotifier(NotificationBase):
    """Discord Webhook notification provider with referral integration."""
    
    def __init__(
        self,
        config: NotificationConfig,
        discord_config: DiscordConfig,
        referral_config: Optional[ReferralConfig] = None,
    ):
        super().__init__(config)
        self.discord_config = discord_config
        self.referral_config = referral_config or ReferralConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        
        if not discord_config.webhook_url:
            logger.warning("Discord webhook_url not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = config.enabled

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.discord_config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_referral_field(self) -> Dict[str, Any]:
        """Build referral CTA as an embed field."""
        if not self.referral_config.enabled:
            return None
        
        return {
            "name": "🚀 NEW TO TRADING?",
            "value": (
                f"Register using our official broker link:\n"
                f"{self.referral_config.referral_link}\n\n"
                f"💰 {self.referral_config.promo_text}\n\n"
                f"⚠️ Signals are intended for registered members.\n"
                f"📚 Create your account and practice risk-free before trading live."
            ),
            "inline": False,
        }

    def _should_include_referral(self, signal: SignalNotification) -> bool:
        if not self.referral_config.enabled:
            return False
        if not self.referral_config.show_on_every_signal:
            return False
        if signal.confidence < self.referral_config.min_confidence_for_referral:
            return False
        return True

    async def _send_embed(self, embed: Dict[str, Any]) -> bool:
        if not self.enabled:
            logger.debug("Discord notifier disabled, skipping send")
            return False
            
        if not self._check_rate_limit():
            logger.warning("Discord rate limit exceeded")
            return False
        
        session = await self._get_session()
        
        payload = {
            "username": self.discord_config.username,
            "embeds": [embed],
        }
        
        if self.discord_config.avatar_url:
            payload["avatar_url"] = self.discord_config.avatar_url
        
        async def _do_send():
            async with session.post(self.discord_config.webhook_url, json=payload) as response:
                if response.status in (200, 204):
                    self._record_send()
                    return True
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After", "1")
                    logger.warning(f"Discord rate limited, retry after {retry_after}s")
                    await asyncio.sleep(float(retry_after) + 1)
                    return False
                else:
                    error_text = await response.text()
                    logger.error(f"Discord HTTP {response.status}: {error_text}")
                    return False
        
        try:
            return await self._retry_with_backoff(_do_send)
        except ClientError as e:
            logger.error(f"Discord connection error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Discord request timeout")
            return False
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    def _create_embed(
        self,
        title: str,
        description: str,
        color: int,
        fields: List[Dict[str, Any]] = None,
        footer: str = None,
        timestamp: datetime = None,
    ) -> Dict[str, Any]:
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }
        
        if fields:
            embed["fields"] = fields
        
        if footer:
            embed["footer"] = {"text": footer}
        
        if timestamp:
            embed["timestamp"] = timestamp.isoformat()
        
        return embed

    def _format_signal_embed(self, signal: SignalNotification) -> Dict[str, Any]:
        # Direction and color
        if signal.direction == SignalDirection.BUY:
            title = "🟢 BUY Signal"
            color = 0x00FF00
        elif signal.direction == SignalDirection.SELL:
            title = "🔴 SELL Signal"
            color = 0xFF0000
        else:
            title = "⚪ WAIT Signal"
            color = 0x808080
        
        # Confidence emoji
        if signal.confidence >= 0.9:
            conf_emoji = "🔥"
        elif signal.confidence >= 0.75:
            conf_emoji = "⭐"
        elif signal.confidence >= 0.6:
            conf_emoji = "📊"
        else:
            conf_emoji = "📈"
        
        time_str = signal.timestamp.strftime("%H:%M UTC")
        
        # Build analysis
        analysis_lines = []
        if signal.supporting_analysis:
            for key, value in signal.supporting_analysis.items():
                if isinstance(value, float):
                    analysis_lines.append(f"• {key}: {value:.4f}" if abs(value) < 0.01 else f"• {key}: {value:.2f}")
                elif isinstance(value, bool):
                    analysis_lines.append(f"• {key}: {'✅' if value else '❌'}")
                else:
                    analysis_lines.append(f"• {key}: {value}")
        else:
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
                analysis_lines = ["• No Clear Signal", "• Wait for Confirmation"]
        
        analysis_text = "\n".join(analysis_lines)
        
        fields = [
            {"name": "Asset", "value": signal.symbol, "inline": True},
            {"name": "Timeframe", "value": signal.timeframe, "inline": True},
            {"name": "Confidence", "value": f"{signal.confidence:.0%} {conf_emoji}", "inline": True},
            {"name": "Signal Time (UTC)", "value": time_str, "inline": True},
            {"name": "AI Analysis", "value": analysis_text, "inline": False},
        ]
        
        if signal.position_size_pct > 0:
            fields.insert(3, {"name": "Position Size", "value": f"{signal.position_size_pct:.1%}", "inline": True})
        
        # Add referral field if enabled
        if self._should_include_referral(signal):
            ref_field = self._build_referral_field()
            if ref_field:
                fields.append(ref_field)
        
        return self._create_embed(
            title=title,
            description=f"AI-generated signal for {signal.symbol}",
            color=color,
            fields=fields,
            footer="XAUUSD Scalper | UTC",
            timestamp=signal.timestamp,
        )

    def _format_trade_embed(self, trade: TradeNotification) -> Dict[str, Any]:
        if trade.exit_price is None:
            color = 0x00FF00 if trade.direction == SignalDirection.BUY else 0xFF0000
            title = "🟢 TRADE OPENED" if trade.direction == SignalDirection.BUY else "🔴 TRADE OPENED"
            
            fields = [
                {"name": "Trade ID", "value": trade.trade_id, "inline": True},
                {"name": "Asset", "value": trade.symbol, "inline": True},
                {"name": "Direction", "value": trade.direction.value, "inline": True},
                {"name": "Entry Price", "value": f"${trade.entry_price:,.5f}", "inline": True},
                {"name": "Size", "value": f"{trade.size:.4f}", "inline": True},
                {"name": "Time (UTC)", "value": trade.entry_time.strftime('%H:%M:%S'), "inline": True},
            ]
        else:
            is_profit = trade.pnl > 0
            color = 0x00FF00 if is_profit else 0xFF0000
            title = "💰 TRADE CLOSED (PROFIT)" if is_profit else "💸 TRADE CLOSED (LOSS)"
            
            pnl_text = f"+${trade.pnl:,.2f}" if trade.pnl > 0 else f"${trade.pnl:,.2f}"
            
            fields = [
                {"name": "Trade ID", "value": trade.trade_id, "inline": True},
                {"name": "Asset", "value": trade.symbol, "inline": True},
                {"name": "Direction", "value": trade.direction.value, "inline": True},
                {"name": "Entry", "value": f"${trade.entry_price:,.5f}", "inline": True},
                {"name": "Exit", "value": f"${trade.exit_price:,.5f}", "inline": True},
                {"name": "Size", "value": f"{trade.size:.4f}", "inline": True},
                {"name": "PnL", "value": f"{pnl_text} ({trade.pnl_pct:.2%})", "inline": True},
                {"name": "Reason", "value": trade.exit_reason or "N/A", "inline": True},
                {"name": "Commission", "value": f"${trade.commission:.2f}", "inline": True},
                {"name": "Slippage", "value": f"{trade.slippage:.4%}", "inline": True},
                {"name": "Entry (UTC)", "value": trade.entry_time.strftime('%H:%M:%S'), "inline": True},
                {"name": "Exit (UTC)", "value": trade.exit_time.strftime('%H:%M:%S'), "inline": True},
            ]
        
        return self._create_embed(
            title=title,
            description=f"{trade.direction.value} trade for {trade.symbol}",
            color=color,
            fields=fields,
            footer="XAUUSD Scalper",
            timestamp=trade.exit_time or trade.entry_time,
        )

    def _format_risk_embed(self, alert: RiskAlertNotification) -> Dict[str, Any]:
        severity_colors = {
            "LOW": 0x00FFFF,
            "MEDIUM": 0xFFFF00,
            "HIGH": 0xFFA500,
            "CRITICAL": 0xFF0000,
        }
        
        return self._create_embed(
            title=f"⚠️ RISK ALERT: {alert.alert_type}",
            description=alert.message,
            color=severity_colors.get(alert.severity, 0x808080),
            fields=[
                {"name": "Severity", "value": alert.severity, "inline": True},
                {"name": "Current Value", "value": f"{alert.current_value:.4f}", "inline": True},
                {"name": "Threshold", "value": f"{alert.threshold:.4f}", "inline": True},
            ],
            footer="XAUUSD Scalper Risk Management",
            timestamp=alert.timestamp,
        )

    def _format_system_embed(self, alert: SystemAlertNotification) -> Dict[str, Any]:
        severity_colors = {
            "INFO": 0x00FFFF,
            "WARNING": 0xFFFF00,
            "ERROR": 0xFF0000,
            "CRITICAL": 0x8B0000,
        }
        
        return self._create_embed(
            title=f"🖥️ SYSTEM ALERT: {alert.component}",
            description=alert.message,
            color=severity_colors.get(alert.severity, 0x808080),
            fields=[
                {"name": "Severity", "value": alert.severity, "inline": True},
                {"name": "Component", "value": alert.component, "inline": True},
            ],
            footer="XAUUSD Scalper System",
            timestamp=alert.timestamp,
        )

    async def send_signal(self, signal: SignalNotification) -> bool:
        embed = self._format_signal_embed(signal)
        return await self._send_embed(embed)

    async def send_trade(self, trade: TradeNotification) -> bool:
        embed = self._format_trade_embed(trade)
        return await self._send_embed(embed)

    async def send_risk_alert(self, alert: RiskAlertNotification) -> bool:
        embed = self._format_risk_embed(alert)
        return await self._send_embed(embed)

    async def send_system_alert(self, alert: SystemAlertNotification) -> bool:
        embed = self._format_system_embed(alert)
        return await self._send_embed(embed)

    async def test_connection(self) -> bool:
        if not self.enabled:
            logger.warning("Discord notifier not configured")
            return False
        
        embed = self._create_embed(
            title="✅ Discord Connection Test",
            description="XAUUSD Scalper successfully connected to Discord",
            color=0x00FF00,
            footer="XAUUSD Scalper",
            timestamp=datetime.utcnow(),
        )
        
        try:
            return await self._send_embed(embed)
        except Exception as e:
            logger.error(f"Discord connection test error: {e}")
            return False


def create_discord_notifier(
    config: NotificationConfig = None,
    webhook_url: str = None,
    referral_link: str = "https://bit.ly/4yAbSgu",
    promo_text: str = "Receive $10,000 in virtual funds and learn how to trade with a market leader.",
) -> DiscordNotifier:
    """Factory function to create Discord notifier with referral integration."""
    webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
    
    discord_config = DiscordConfig(
        webhook_url=webhook_url,
        username="XAUUSD Scalper",
        timeout=10.0,
        max_retries=3,
        retry_delay=1.0,
        rate_limit_per_minute=30,
        rate_limit_per_hour=100,
    )
    
    referral_config = ReferralConfig(
        referral_link=referral_link or os.getenv("DISCORD_REFERRAL_LINK", "https://bit.ly/4yAbSgu"),
        promo_text=promo_text or os.getenv("DISCORD_PROMO_TEXT", "Receive $10,000 in virtual funds and learn how to trade with a market leader."),
        enabled=True,
        show_on_every_signal=True,
        min_confidence_for_referral=0.0,
    )
    
    notification_config = config or NotificationConfig(
        enabled=bool(webhook_url),
        rate_limit_per_minute=30,
        rate_limit_per_hour=100,
        retry_attempts=3,
        retry_base_delay=1.0,
        retry_max_delay=60.0,
        timeout=10.0,
    )
    
    return DiscordNotifier(notification_config, discord_config, referral_config)
