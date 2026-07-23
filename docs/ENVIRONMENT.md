# Environment Variables

This document describes all environment variables used by the XAUUSD AI Scalping System.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYTHONPATH` | Yes | `/app` | Python module search path |
| `CONFIG_PATH` | Yes | `config/config.yaml` | Path to configuration file |
| `PORT` | Auto | `10000` | Render assigns this automatically |
| `TELEGRAM_BOT_TOKEN` | No | - | Telegram bot token |
| `TELEGRAM_CHAT_ID` | No | - | Telegram chat ID for notifications |
| `TELEGRAM_ENABLED` | No | `false` | Enable Telegram notifications |
| `API_HOST` | No | `0.0.0.0` | API bind address |
| `API_PORT` | No | `8000` | API port (use PORT on Render) |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOG_JSON_FORMAT` | No | `false` | JSON log format |
| `DATA_SYMBOL` | No | `GC=F` | Trading symbol |
| `DATA_INTERVAL` | No | `1m` | Data interval |

---

## Required Variables

### PYTHONPATH
- **Required**: Yes
- **Default**: `/app`
- **Description**: Python module search path. Must include the directory containing the `src` package.
- **Render**: Set to `/app`

### CONFIG_PATH
- **Required**: Yes
- **Default**: `config/config.yaml`
- **Description**: Path to the YAML configuration file
- **Render**: `config/config.yaml` (relative to app directory)

### PORT
- **Required**: Auto (set by Render)
- **Description**: Port for the web server. Render sets this automatically.
- **Render**: Automatically assigned (typically 10000)

---

## Telegram Configuration

### TELEGRAM_BOT_TOKEN
- **Required**: No (required for notifications)
- **Description**: Telegram Bot Token from BotFather
- **Example**: `123456789:ABCdefGHIjklMNOpqrSTUvwxyz`
- **Where to get**: Message @BotFather on Telegram

### TELEGRAM_CHAT_ID
- **Required**: No (required for notifications)
- **Description**: Telegram Chat ID to send notifications to
- **Example**: `-1001234567890` or `123456789`
- **Where to get**: Message @userinfobot or check channel ID

### TELEGRAM_ENABLED
- **Required**: No
- **Default**: `false`
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`
- **Description**: Enable/disable Telegram notifications

### TELEGRAM_PARSE_MODE
- **Required**: No
- **Default**: `Markdown`
- **Values**: `Markdown`, `HTML`
- **Description**: Telegram message format

### TELEGRAM_TIMEOUT
- **Required**: No
- **Default**: `10.0`
- **Description**: Telegram API timeout in seconds

### TELEGRAM_MAX_RETRIES
- **Required**: No
- **Default**: `3`
- **Description**: Maximum retry attempts for failed messages

### TELEGRAM_REFERRAL_LINK
- **Required**: No
- **Default**: `https://bit.ly/4yAbSgu`
- **Description**: Exness referral link included in notifications

---

## API Configuration

### API_HOST
- **Required**: No
- **Default**: `0.0.0.0`
- **Description**: Host address to bind the API server
- **Render**: `0.0.0.0`

### API_PORT
- **Required**: No (use PORT on Render)
- **Default**: `8000`
- **Description**: Port for the API server
- **Render**: Use the `PORT` environment variable instead

### API_WORKERS
- **Required**: No
- **Default**: `4`
- **Description**: Number of worker processes

### API_TIMEOUT
- **Required**: No
- **Default**: `30`
- **Description**: Request timeout in seconds

### API_AUTH_ENABLED
- **Required**: No
- **Default**: `false`
- **Description**: Enable API key authentication

### API_KEY
- **Required**: No
- **Description**: API key for authentication (if enabled)

### API_RATE_LIMIT
- **Required**: No
- **Default**: `100`
- **Description**: Requests per minute rate limit

### API_CORS_ORIGINS
- **Required**: No
- **Default**: `*`
- **Description**: Allowed CORS origins (comma-separated)

---

## Logging Configuration

### LOG_LEVEL
- **Required**: No
- **Default**: `INFO`
- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Description**: Logging level for the application

### LOG_FORMAT
- **Required**: No
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Description**: Python logging format string

### LOG_JSON_FORMAT
- **Required**: No
- **Default**: `false`
- **Values**: `true`, `false`
- **Description**: Use JSON format for logs (good for log aggregators)

### LOG_FILE
- **Required**: No
- **Default**: `logs/xauusd_scalper.log`
- **Description**: Path to log file

### LOG_MAX_BYTES
- **Required**: No
- **Default**: `10485760` (10MB)
- **Description**: Maximum log file size before rotation

### LOG_BACKUP_COUNT
- **Required**: No
- **Default**: `10`
- **Description**: Number of backup log files to keep

---

## Data Source Configuration

### DATA_SYMBOL
- **Required**: No
- **Default**: `GC=F`
- **Description**: Yahoo Finance ticker symbol for gold
- **Options**:
  - `GC=F` - Gold Futures
  - `XAUUSD=X` - XAU/USD Spot
  - `GCM25.CMX` - Gold Futures (CMX)

### DATA_INTERVAL
- **Required**: No
- **Default**: `1m`
- **Description**: Default data interval
- **Options**: `1m`, `2m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1wk`, `1mo`

### DATA_PERIOD
- **Required**: No
- **Default**: `7d`
- **Description**: Default data period
- **Options**: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

### DATA_CACHE_DIR
- **Required**: No
- **Default**: `data/cache`
- **Description**: Directory for cached data

### DATA_CACHE_TTL
- **Required**: No
- **Default**: `5`
- **Description**: Cache time-to-live in minutes

### DATA_MAX_RETRIES
- **Required**: No
- **Default**: `3`
- **Description**: Maximum retry attempts for data fetch failures

### DATA_RETRY_DELAY
- **Required**: No
- **Default**: `5`
- **Description**: Delay between retries in seconds

---

## Signal Generation Configuration

### SIGNAL_CHECK_INTERVAL
- **Required**: No (worker only)
- **Default**: `60`
- **Description**: Interval between signal checks in seconds

### SIGNAL_CONFIDENCE_THRESHOLD
- **Required**: No
- **Default**: `0.55`
- **Description**: Minimum confidence for signal generation

### SIGNAL_MIN_RISK_REWARD
- **Required**: No
- **Default**: `1.5`
- **Description**: Minimum risk/reward ratio

### SIGNAL_TREND_FILTER
- **Required**: No
- **Default**: `true`
- **Description**: Enable trend filter

### SIGNAL_VOL_FILTER
- **Required**: No
- **Default**: `true`
- **Description**: Enable volatility filter

### SIGNAL_VOLUME_FILTER
- **Required**: No
- **Default**: `true`
- **Description**: Enable volume filter

### SIGNAL_SESSION_FILTER
- **Required**: No
- **Default**: `true`
- **Description**: Enable trading session filter

### SIGNAL_ALLOWED_SESSIONS
- **Required**: No
- **Default**: `london,new_york,overlap`
- **Description**: Allowed trading sessions (comma-separated)

---

## Model Configuration

### MODEL_DIR
- **Required**: No
- **Default**: `models`
- **Description**: Directory for trained models

### MODEL_TEST_SIZE
- **Required**: No
- **Default**: `0.2`
- **Description**: Test set size for training

### MODEL_DEFAULT
- **Required**: No
- **Default**: `ensemble`
- **Description**: Default model name

### MODEL_SAVE_FEATURE_IMPORTANCE
- **Required**: No
- **Default**: `true`
- **Description**: Save feature importance after training

---

## Application Configuration

### ENVIRONMENT
- **Required**: No
- **Default**: `production`
- **Values**: `development`, `production`, `testing`
- **Description**: Application environment

### APP_DEBUG
- **Required**: No
- **Default**: `false`
- **Description**: Enable debug mode

### APP_ENV
- **Required**: No
- **Default**: `production`
- **Description**: Alias for ENVIRONMENT

---

## Render-Specific Variables

These are automatically set by Render:

| Variable | Description |
|----------|-------------|
| `PORT` | Assigned port for the web service |
| `RENDER_SERVICE_NAME` | Name of the Render service |
| `RENDER_DEPLOYMENT_ID` | Unique deployment identifier |
| `RENDER_INSTANCE_ID` | Instance identifier |

---

## Local Development Setup

Create a `.env` file in the project root:

```bash
# Required
PYTHONPATH=/path/to/project
CONFIG_PATH=config/config.yaml

# Optional - Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ENABLED=false

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=DEBUG
LOG_JSON_FORMAT=false

# Data
DATA_SYMBOL=GC=F
DATA_INTERVAL=1m

# Model
MODEL_DIR=models
```

---

## Docker Deployment

```bash
# Build and run with environment
docker build -t xauusd-scalper .
docker run -d \
  -p 8000:8000 \
  -e PYTHONPATH=/app \
  -e CONFIG_PATH=config/config.yaml \
  -e PORT=8000 \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  xauusd-scalper
```

---

## Production Checklist

Before deploying to production, ensure:

- [ ] `PYTHONPATH` is set correctly (`/app` on Render)
- [ ] `CONFIG_PATH` points to a valid config file
- [ ] `LOG_LEVEL` is set to `INFO` or `WARNING`
- [ ] `LOG_JSON_FORMAT` is `true` for log aggregation
- [ ] Telegram credentials are set (if notifications enabled)
- [ ] `API_AUTH_ENABLED` and `API_KEY` are set (if securing API)
- [ ] `DATA_SYMBOL` matches desired trading symbol
- [ ] `MODEL_DIR` points to persistent storage for model persistence
