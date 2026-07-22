"""Application lifecycle management for XAUUSD Scalping System."""
import asyncio
import signal
import logging
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """Application lifecycle states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LifecycleHook:
    """Lifecycle hook definition."""
    name: str
    callback: Callable[[], Awaitable[None]]
    priority: int = 0
    phase: str = ""  # startup, shutdown, health_check


class LifecycleManager:
    """Manages application lifecycle with graceful startup/shutdown."""
    
    def __init__(
        self,
        app_name: str = "XAUUSD Scalper",
        shutdown_timeout: float = 30.0,
        startup_timeout: float = 60.0,
    ):
        self.app_name = app_name
        self.shutdown_timeout = shutdown_timeout
        self.startup_timeout = startup_timeout
        
        self._state = ApplicationState.INITIALIZING
        self._hooks: Dict[str, List[LifecycleHook]] = {
            "startup": [],
            "shutdown": [],
            "health_check": [],
        }
        self._startup_complete = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        self._shutdown_requested = False
        self._start_time: Optional[datetime] = None
        self._health_data: Dict[str, Any] = {}
        self._signal_handlers_installed = False

    @property
    def state(self) -> ApplicationState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == ApplicationState.RUNNING

    @property
    def is_stopping(self) -> bool:
        return self._state in (ApplicationState.STOPPING, ApplicationState.STOPPED)

    @property
    def uptime(self) -> float:
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return 0.0

    def add_startup_hook(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add a startup hook. Higher priority runs first."""
        hook = LifecycleHook(name=name, callback=callback, priority=priority, phase="startup")
        self._hooks["startup"].append(hook)
        self._hooks["startup"].sort(key=lambda h: -h.priority)
        logger.debug(f"Added startup hook: {name} (priority: {priority})")

    def add_shutdown_hook(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add a shutdown hook. Higher priority runs first."""
        hook = LifecycleHook(name=name, callback=callback, priority=priority, phase="shutdown")
        self._hooks["shutdown"].append(hook)
        self._hooks["shutdown"].sort(key=lambda h: -h.priority)
        logger.debug(f"Added shutdown hook: {name} (priority: {priority})")

    def add_health_check_hook(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add a health check hook."""
        hook = LifecycleHook(name=name, callback=callback, priority=priority, phase="health_check")
        self._hooks["health_check"].append(hook)
        self._hooks["health_check"].sort(key=lambda h: -h.priority)
        logger.debug(f"Added health check hook: {name} (priority: {priority})")

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return
            
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)
            self._signal_handlers_installed = True
            logger.debug("Signal handlers installed")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not install signal handlers: {e}")

    async def startup(self) -> bool:
        """Run startup sequence."""
        if self._state != ApplicationState.INITIALIZING:
            logger.warning(f"Startup called in invalid state: {self._state}")
            return False
        
        self._state = ApplicationState.STARTING
        self._start_time = datetime.utcnow()
        logger.info(f"{self.app_name} starting up...")
        
        try:
            await asyncio.wait_for(
                self._run_hooks("startup"),
                timeout=self.startup_timeout
            )
            
            self._state = ApplicationState.RUNNING
            self._startup_complete.set()
            logger.info(f"{self.app_name} startup complete in {self.uptime:.2f}s")
            return True
            
        except asyncio.TimeoutError:
            self._state = ApplicationState.ERROR
            logger.error(f"Startup timeout after {self.startup_timeout}s")
            return False
        except Exception as e:
            self._state = ApplicationState.ERROR
            logger.exception(f"Startup failed: {e}")
            return False

    async def shutdown(self, reason: str = "Requested") -> bool:
        """Run shutdown sequence."""
        if self._shutdown_requested:
            logger.debug("Shutdown already in progress")
            return True
            
        if self._state == ApplicationState.STOPPED:
            logger.debug("Already stopped")
            return True
        
        self._shutdown_requested = True
        self._state = ApplicationState.STOPPING
        logger.info(f"{self.app_name} shutting down: {reason}")
        
        try:
            await asyncio.wait_for(
                self._run_hooks("shutdown"),
                timeout=self.shutdown_timeout
            )
            
            self._state = ApplicationState.STOPPED
            self._shutdown_complete.set()
            logger.info(f"{self.app_name} shutdown complete")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Shutdown timeout after {self.shutdown_timeout}s")
            self._state = ApplicationState.ERROR
            return False
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
            self._state = ApplicationState.ERROR
            return False

    async def _run_hooks(self, phase: str):
        """Run all hooks for a phase."""
        hooks = self._hooks.get(phase, [])
        if not hooks:
            logger.debug(f"No {phase} hooks registered")
            return
        
        logger.info(f"Running {len(hooks)} {phase} hooks...")
        
        for hook in hooks:
            try:
                logger.debug(f"Running {phase} hook: {hook.name}")
                await hook.callback()
                logger.debug(f"Completed {phase} hook: {hook.name}")
            except Exception as e:
                logger.exception(f"Hook {hook.name} failed: {e}")
                raise

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health check hooks."""
        health_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": self.uptime,
            "state": self._state.value,
            "checks": {}
        }
        
        hooks = self._hooks.get("health_check", [])
        for hook in hooks:
            try:
                result = await hook.callback()
                health_data["checks"][hook.name] = {
                    "status": "healthy",
                    "result": result,
                }
            except Exception as e:
                health_data["checks"][hook.name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                logger.warning(f"Health check {hook.name} failed: {e}")
        
        self._health_data = health_data
        return health_data

    def get_health_data(self) -> Dict[str, Any]:
        """Get cached health data."""
        return self._health_data

    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for application lifespan."""
        success = await self.startup()
        if not success:
            raise RuntimeError("Application startup failed")
        
        self.install_signal_handlers()
        
        try:
            yield self
        finally:
            await self.shutdown("Context exit")

    async def wait_for_startup(self, timeout: float = None) -> bool:
        """Wait for startup to complete."""
        try:
            await asyncio.wait_for(
                self._startup_complete.wait(),
                timeout=timeout or self.startup_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_shutdown(self, timeout: float = None) -> bool:
        """Wait for shutdown to complete."""
        try:
            await asyncio.wait_for(
                self._shutdown_complete.wait(),
                timeout=timeout or self.shutdown_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False


class GracefulShutdown:
    """Context manager for graceful shutdown handling."""
    
    def __init__(self, lifecycle: LifecycleManager):
        self.lifecycle = lifecycle
        self._original_handlers: Dict[int, Any] = {}

    async def __aenter__(self) -> LifecycleManager:
        self.lifecycle.install_signal_handlers()
        return self.lifecycle

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lifecycle.shutdown("Context exit")


class Application:
    """Main application wrapper with lifecycle management."""
    
    def __init__(self, name: str = "XAUUSD Scalper", settings: Any = None):
        self.name = name
        self.settings = settings
        self.lifecycle = LifecycleManager(app_name=name)
        self._running = False

    def add_startup(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add startup hook."""
        self.lifecycle.add_startup_hook(name, callback, priority)

    def add_shutdown(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add shutdown hook."""
        self.lifecycle.add_shutdown_hook(name, callback, priority)

    def add_health_check(self, name: str, callback: Callable[[], Awaitable[None]], priority: int = 0):
        """Add health check hook."""
        self.lifecycle.add_health_check_hook(name, callback, priority)

    async def start(self) -> bool:
        """Start the application."""
        if self._running:
            logger.warning("Application already running")
            return True
        
        self._running = True
        return await self.lifecycle.startup()

    async def stop(self, reason: str = "Requested") -> bool:
        """Stop the application."""
        if not self._running:
            logger.warning("Application not running")
            return True
        
        self._running = False
        return await self.lifecycle.shutdown(reason)

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks."""
        return await self.lifecycle.run_health_checks()

    @property
    def state(self) -> ApplicationState:
        return self.lifecycle.state

    @property
    def is_running(self) -> bool:
        return self.lifecycle.is_running

    @property
    def uptime(self) -> float:
        return self.lifecycle.uptime

    async def __aenter__(self) -> 'Application':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop("Context exit")


# Global application instance
_app: Optional[Application] = None


def get_application() -> Application:
    """Get or create global application instance."""
    global _app
    if _app is None:
        _app = Application()
    return _app


def set_application(app: Application):
    """Set global application instance."""
    global _app
    _app = app
