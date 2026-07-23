"""Health check system for XAUUSD Scalping System."""
import asyncio
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "timestamp": c.timestamp.isoformat(),
                    "latency_ms": c.latency_ms,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Manages health checks for all components."""
    
    def __init__(self, app_name: str = "XAUUSD Scalper"):
        self.app_name = app_name
        self._checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self._start_time = datetime.utcnow()
        self._last_check: Optional[SystemHealth] = None

    def register_check(self, name: str, check_func: Callable[[], Awaitable[ComponentHealth]]):
        """Register a health check function."""
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            logger.debug(f"Unregistered health check: {name}")

    async def run_check(self, name: str) -> ComponentHealth:
        """Run a single health check."""
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not registered",
            )
        
        start_time = time.perf_counter()
        try:
            result = await self._checks[name]()
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Health check {name} failed: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency_ms,
            )

    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        components = []
        
        # Run checks concurrently
        tasks = {
            name: asyncio.create_task(self.run_check(name))
            for name in self._checks
        }
        
        for name, task in tasks.items():
            try:
                result = await task
                components.append(result)
            except Exception as e:
                logger.exception(f"Health check {name} task failed: {e}")
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                ))
        
        # Determine overall status
        statuses = [c.status for c in components]
        if not statuses:
            overall_status = HealthStatus.UNKNOWN
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        health = SystemHealth(
            status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            uptime_seconds=(datetime.utcnow() - self._start_time).total_seconds(),
        )
        
        self._last_check = health
        return health

    def get_last_check(self) -> Optional[SystemHealth]:
        """Get the last health check result."""
        return self._last_check

    def is_healthy(self) -> bool:
        """Check if system is healthy based on last check."""
        if self._last_check is None:
            return False
        return self._last_check.status == HealthStatus.HEALTHY


# Pre-defined health checks
async def check_database() -> ComponentHealth:
    """Check database connectivity (placeholder)."""
    # Implement actual database check
    return ComponentHealth(
        name="database",
        status=HealthStatus.HEALTHY,
        message="Database connection OK",
    )


async def check_telegram() -> ComponentHealth:
    """Check Telegram Bot connectivity."""
    try:
        from notifications.telegram import create_telegram_notifier
        from notifications.base import NotificationConfig
        
        notifier = create_telegram_notifier()
        if not notifier.enabled:
            return ComponentHealth(
                name="telegram",
                status=HealthStatus.DEGRADED,
                message="Telegram not configured",
            )
        
        connected = await notifier.test_connection()
        await notifier.close()
        
        if connected:
            return ComponentHealth(
                name="telegram",
                status=HealthStatus.HEALTHY,
                message="Telegram Bot connected",
            )
        else:
            return ComponentHealth(
                name="telegram",
                status=HealthStatus.UNHEALTHY,
                message="Telegram Bot connection failed",
            )
    except Exception as e:
        return ComponentHealth(
            name="telegram",
            status=HealthStatus.UNHEALTHY,
            message=f"Telegram check error: {e}",
        )


async def check_data_source() -> ComponentHealth:
    """Check data source availability."""
    try:
        from src.data.yahoo_client import YahooFinanceClient
        
        client = YahooFinanceClient()
        price = await asyncio.get_event_loop().run_in_executor(None, client.get_current_price)
        
        if price is not None:
            return ComponentHealth(
                name="data_source",
                status=HealthStatus.HEALTHY,
                message=f"Data source OK (price: ${price:.2f})",
                details={"current_price": price},
            )
        else:
            return ComponentHealth(
                name="data_source",
                status=HealthStatus.DEGRADED,
                message="Data source returned no price",
            )
    except Exception as e:
        return ComponentHealth(
            name="data_source",
            status=HealthStatus.UNHEALTHY,
            message=f"Data source check failed: {e}",
        )


async def check_model() -> ComponentHealth:
    """Check model availability."""
    try:
        from src.models.persistence import ModelPersistence
        
        persistence = ModelPersistence()
        models = persistence.list_models()
        
        if models:
            return ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
                message=f"Models available: {len(models)}",
                details={"models": models},
            )
        else:
            return ComponentHealth(
                name="model",
                status=HealthStatus.DEGRADED,
                message="No models trained",
            )
    except Exception as e:
        return ComponentHealth(
            name="model",
            status=HealthStatus.UNHEALTHY,
            message=f"Model check failed: {e}",
        )


async def check_disk_space() -> ComponentHealth:
    """Check available disk space."""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage("/")
        free_pct = (free / total) * 100
        
        if free_pct > 20:
            status = HealthStatus.HEALTHY
        elif free_pct > 10:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return ComponentHealth(
            name="disk_space",
            status=status,
            message=f"Disk space: {free_pct:.1f}% free",
            details={
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_pct": free_pct,
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message=f"Disk check failed: {e}",
        )


async def check_memory() -> ComponentHealth:
    """Check memory usage."""
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        
        if mem.percent < 80:
            status = HealthStatus.HEALTHY
        elif mem.percent < 90:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return ComponentHealth(
            name="memory",
            status=status,
            message=f"Memory usage: {mem.percent:.1f}%",
            details={
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_pct": mem.percent,
            },
        )
    except ImportError:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message=f"Memory check failed: {e}",
        )


def create_default_health_checker(app_name: str = "XAUUSD Scalper") -> HealthChecker:
    """Create a HealthChecker with default checks."""
    checker = HealthChecker(app_name)
    
    # Register default checks
    checker.register_check("data_source", check_data_source)
    checker.register_check("telegram", check_telegram)
    checker.register_check("model", check_model)
    checker.register_check("disk_space", check_disk_space)
    checker.register_check("memory", check_memory)
    
    return checker
