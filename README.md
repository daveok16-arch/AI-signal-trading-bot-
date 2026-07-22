# XAUUSD AI Scalping Signal System

A production-ready, AI-powered scalping signal system for XAUUSD (Gold/USD) trading. Combines machine learning, technical analysis, price action, market structure, volatility, momentum, and volume analysis with ensemble techniques to generate high-confidence BUY, SELL, or WAIT signals for short-term trading.

## Features

### 🤖 Machine Learning
- **Multiple Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Logistic Regression, MLP
- **Ensemble Methods**: Voting, Stacking, Blending with configurable weights
- **Walk-Forward Validation**: Time-series aware cross-validation with purging/embargo
- **Feature Selection**: Mutual information, F-statistic, Random Forest importance, L1 regularization

### 📊 Feature Engineering (200+ Features)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX, OBV, MFI, VWAP, Supertrend, Ichimoku, Keltner, Donchian, Pivot Points
- **Price Action**: Pinbar, Engulfing, Doji, Hammer, Shooting Star, Morning/Evening Star, Three Soldiers/Crows, Inside/Outside Bar
- **Market Structure**: Swing Highs/Lows, Break of Structure (BOS), Change of Character (CHoCH), Fair Value Gaps (FVG), Order Blocks, Liquidity
- **Volatility**: ATR, NATR, Keltner, Donchian, Historical Vol, Parkinson, Garman-Klass, Yang-Zhang
- **Momentum**: ROC, MOM, TSI, Ultimate Oscillator, KST, Stochastic RSI, CMO, Inertia
- **Volume**: OBV, VWAP, MFI, CMF, EOM, NVI, PVI, VPT, VWMA, Volume Oscillator

### 🛡️ Risk Management
- Position sizing: Fixed, Kelly Criterion, Volatility-adjusted
- Stop loss / Take profit with trailing stops
- Maximum daily trades, daily loss, drawdown limits
- Session filtering (London, New York, Overlap, Asian)
- Spread and volatility filters

### 🔄 Backtesting
- Event-driven engine with realistic execution
- Commission, slippage, spread modeling
- Comprehensive metrics: Sharpe, Sortino, Calmar, Profit Factor, Expectancy, VaR, CVaR
- Walk-forward analysis with expanding/rolling windows

### 📈 Real-time Dashboard
- Streamlit-based interactive dashboard
- Live price charts with signals
- Equity curve and drawdown visualization
- Trade analysis and performance metrics
- Auto-refresh capability

### 🌐 REST API
- FastAPI-based service
- Signal generation endpoint
- Historical backtesting endpoint
- Model training endpoint
- Health checks and model management

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/xauusd-scalper.git
cd xauusd-scalper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Configuration

Copy and modify the configuration:

```bash
cp config/config.yaml config/config.local.yaml
# Edit config.local.yaml with your settings
```

### Training a Model

```bash
# Train with default settings
python scripts/train.py

# Train with custom parameters
python scripts/train.py --period 30d --model ensemble --save-model

# Full training with production export
python scripts/train.py --period 60d --model ensemble --save-model --export-production
```

### Running Backtest

```bash
# Backtest last 7 days
python scripts/backtest.py --period 7d --model ensemble

# Backtest specific date range
python scripts/backtest.py --start 2024-01-01 --end 2024-01-31 --model ensemble
```

### Starting the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open http://localhost:8501

### Starting the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API docs available at http://localhost:8000/docs

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Start with training profile
docker-compose --profile training up trainer

# Start with backtest profile
docker-compose --profile backtest up backtest
```

## Project Structure

```
xauusd-scalper/
├── config/
│   └── config.yaml              # Main configuration
├── src/
│   ├── config/                  # Configuration management
│   ├── data/                    # Data acquisition & loading
│   │   ├── yahoo_client.py      # Yahoo Finance client
│   │   ├── cache.py             # Data caching
│   │   └── loader.py            # High-level data loader
│   ├── features/                # Feature engineering
│   │   ├── technical.py         # Technical indicators
│   │   ├── price_action.py      # Price action patterns
│   │   ├── market_structure.py  # Market structure analysis
│   │   ├── volatility.py        # Volatility features
│   │   ├── momentum.py          # Momentum features
│   │   ├── volume.py            # Volume features
│   │   ├── selection.py         # Feature selection
│   │   └── engineer.py          # Main feature pipeline
│   ├── models/                  # Machine learning models
│   │   ├── trainer.py           # Model training
│   │   ├── ensemble.py          # Ensemble methods
│   │   └── persistence.py       # Model saving/loading
│   ├── backtesting/             # Backtesting engine
│   │   ├── engine.py            # Event-driven backtester
│   │   ├── walk_forward.py      # Walk-forward validation
│   │   └── metrics.py           # Performance metrics
│   ├── signals/                 # Signal generation
│   │   └── generator.py         # Signal generator with filters
│   ├── evaluation/              # Performance evaluation
│   │   ├── metrics.py           # Metrics calculation
│   │   └── reporting.py         # Report generation
│   ├── logging/                 # Logging configuration
│   ├── api/                     # REST API
│   │   ├── main.py              # FastAPI app
│   │   └── routes.py            # API routes
│   └── dashboard/               # Streamlit dashboard
│       └── app.py               # Dashboard application
├── scripts/
│   ├── train.py                 # Training script
│   └── backtest.py              # Backtesting script
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── models/                      # Saved models (gitignored)
├── data/                        # Cached data (gitignored)
├── logs/                        # Log files (gitignored)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Docker Compose
└── README.md                    # This file
```

## Configuration

Key configuration sections in `config/config.yaml`:

```yaml
# Data settings
data:
  symbol: "XAUUSD=X"
  interval: "1m"
  period: "7d"
  cache_dir: "data/cache"

# Feature engineering
features:
  technical:
    enabled: true
    indicators:
      - name: "rsi"
        params: [14, 21]
  price_action:
    enabled: true
    patterns: ["pinbar", "engulfing", "doji", ...]
  market_structure:
    enabled: true
    swing_lookback: 10

# Model settings
models:
  ensemble:
    enabled: true
    voting: "soft"
  xgboost:
    enabled: true
    params:
      n_estimators: 500
      max_depth: 6
      learning_rate: 0.01

# Target definition
target:
  type: "classification"
  horizon: 5
  threshold_long: 0.0015
  threshold_short: -0.0015
  labeling_method: "triple_barrier"

# Walk-forward validation
walk_forward:
  enabled: true
  train_window: 5000
  test_window: 1000
  step_size: 500

# Backtesting
backtest:
  initial_capital: 100000
  position_size: 0.1
  commission: 0.0001
  slippage: 0.0001
  stop_loss_pct: 0.005
  take_profit_pct: 0.01

# Signal generation
signals:
  confidence_threshold: 0.55
  min_risk_reward: 1.5
  session_filter:
    enabled: true
    allowed_sessions: ["london", "new_york", "overlap"]
```

## API Reference

### Get Latest Signal
```bash
curl http://localhost:8000/api/v1/signals/latest
```

### Get Signal History
```bash
curl "http://localhost:8000/api/v1/signals/history?hours=24"
```

### Run Backtest
```bash
curl -X POST http://localhost:8000/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{"period": "7d", "model_name": "ensemble"}'
```

### Train Model
```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "ensemble", "period": "30d"}'
```

### List Models
```bash
curl http://localhost:8000/api/v1/models
```

## Performance Metrics

The system tracks comprehensive metrics:

| Category | Metrics |
|----------|---------|
| **Returns** | Total Return, Annualized Return, CAGR |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio |
| **Drawdown** | Max Drawdown, Max DD Duration, Average DD, Ulcer Index |
| **Trades** | Win Rate, Profit Factor, Expectancy, Avg Win/Loss, Risk/Reward |
| **Consistency** | Max Consecutive Wins/Losses, Recovery Factor |
| **Tail Risk** | VaR 95%/99%, CVaR 95%/99%, Tail Ratio, Gain/Pain Ratio |

## Testing

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_features.py::TestTechnicalIndicators::test_rsi -v
```

## Best Practices for Production

1. **Data Quality**: Monitor Yahoo Finance data quality, consider redundant sources
2. **Model Retraining**: Schedule regular retraining (weekly/monthly)
3. **Feature Stability**: Monitor feature importance drift
4. **Risk Limits**: Set conservative limits for live trading
5. **Paper Trading**: Validate signals in paper trading before live
6. **Monitoring**: Set up alerts for model performance degradation

## Disclaimer

⚠️ **This software is for educational and research purposes only. Past performance does not guarantee future results. Trading financial instruments carries significant risk of loss. The authors are not responsible for any financial losses incurred through the use of this system. Always conduct your own due diligence and consult with a qualified financial advisor before making trading decisions.**

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Support

For issues and questions:
- Open a GitHub issue
- Check existing issues and discussions
- Review the documentation

## Acknowledgments

- Yahoo Finance for market data
- scikit-learn, XGBoost, LightGBM, CatBoost communities
- Streamlit and FastAPI teams
- Financial ML community (mlfinlab, etc.)

## 🚀 Render Deployment

This application is designed to run reliably on **Render.com** as a Web Service (API) and Background Worker (Signal Generator).

### Prerequisites

1. Render account
2. Telegram Bot Token (from @BotFather)
3. Telegram Chat ID (from @userinfobot)

### Quick Deploy

1. **Fork this repository** to your GitHub/GitLab
2. **Connect to Render** and create a new "Blueprint" deployment
3. **Set Environment Variables** in Render dashboard:
   - `TELEGRAM_BOT_TOKEN` - Your bot token from @BotFather
   - `TELEGRAM_CHAT_ID` - Your chat ID from @userinfobot
   - `API_KEY` - (Optional) API authentication key
4. **Deploy** - Render will create:
   - **Web Service** (`xauusd-api`) - REST API on port 8000
   - **Background Worker** (`xauusd-signals`) - Signal generation loop

### Services Created

| Service | Type | Purpose |
|---------|------|---------|
| `xauusd-api` | Web Service | REST API for signals, backtesting, model management |
| `xauusd-signals` | Background Worker | Real-time signal generation and Telegram notifications |

### Environment Variables

**Required (Secrets):**
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=-1001234567890
```

**Optional:**
```bash
API_KEY=your-secure-api-key          # Enable API auth
DISCORD_WEBHOOK_URL=https://...      # Discord notifications
NEWS_API_KEY=your-news-api-key       # News filter
```

### API Endpoints

After deployment, your API will be available at:
```
https://xauusd-api.onrender.com
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/signals/latest` | GET | Latest BUY/SELL/WAIT signal |
| `/api/v1/signals/history` | GET | Recent signal history |
| `/api/v1/backtest` | POST | Run backtest |
| `/api/v1/train` | POST | Train new model |
| `/api/v1/models` | GET | List trained models |
| `/api/v1/data/latest` | GET | Latest market data |
| `/docs` | GET | Swagger API docs |

### Telegram Notifications

Once deployed, the system automatically sends:
- 🚀 **Startup notification** when worker starts
- 🟢 **BUY signals** with entry, SL, TP, confidence, analysis
- 🔴 **SELL signals** with entry, SL, TP, confidence, analysis
- 🛑 **Shutdown notification** on graceful stop
- ⚠️ **Risk alerts** for drawdown/position limits

### Health Checks

The application includes comprehensive health checks:
- Database connectivity
- Telegram Bot API
- Data source (Yahoo Finance)
- Model availability
- Disk space & memory

### Graceful Shutdown

The application handles SIGTERM/SIGINT gracefully:
1. Receives shutdown signal
2. Stops accepting new work
3. Finishes current operations
4. Sends shutdown notification to Telegram
5. Closes connections cleanly

### Monitoring

Check service health:
```bash
# Health endpoint
curl https://xauusd-api.onrender.com/health

# Latest signal
curl https://xauusd-api.onrender.com/api/v1/signals/latest
```

### Logs

View logs in Render dashboard or via CLI:
```bash
render logs --service xauusd-api
render logs --service xauusd-signals
```

### Scaling

- **API**: Scale horizontally by increasing plan
- **Worker**: Single instance recommended (avoids duplicate signals)
- **Database**: Add PostgreSQL for persistent storage if needed

### Troubleshooting

**Telegram not working?**
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
- Check bot is added to the chat/group
- Ensure bot has permission to send messages

**Signals not generating?**
- Check worker logs for data fetching errors
- Verify Yahoo Finance is accessible
- Check model files exist in `models/`

**Health checks failing?**
- Check individual component health in `/health` response
- Review logs for specific component errors
