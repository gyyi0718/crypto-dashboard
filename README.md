# 📈 Crypto Trading Dashboard

Real-time cryptocurrency analysis dashboard with technical indicators.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo

**[👉 View Live Dashboard](https://your-app-name.streamlit.app)**

## ✨ Features

- **Real-time Price Tracking**: BTC, ETH, SOL, XRP, DOGE, BNB
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (10/30 period)
  - Bollinger Bands
- **Trading Signals**: Automated buy/sell signal generation
- **Interactive Charts**: Candlestick, volume, indicators
- **Auto Refresh**: Real-time updates every 10 seconds

## 📸 Screenshots

| Market Overview | Technical Analysis |
|:---:|:---:|
| 6개 심볼 실시간 가격 | 캔들차트 + 지표 |

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

Open http://localhost:8501

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → Deploy

## 📊 Technical Indicators

| Indicator | Description | Signal |
|-----------|-------------|--------|
| **RSI** | Relative Strength Index | < 30: Oversold (Buy), > 70: Overbought (Sell) |
| **MACD** | Trend Momentum | Bullish/Bearish crossover |
| **EMA** | Exponential Moving Average | 10 > 30: Uptrend, 10 < 30: Downtrend |
| **Bollinger** | Volatility Bands | Position within bands |

## 🔧 Configuration

Edit `streamlit_app.py`:

```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]  # Add/remove symbols
```

## 📁 Project Structure

```
crypto-dashboard/
├── streamlit_app.py     # Main application
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Charts**: Plotly
- **Data**: Binance Futures API
- **Hosting**: Streamlit Cloud (Free)

## ⚠️ Disclaimer

This dashboard is for **educational purposes only**. 

- Not financial advice
- Do your own research (DYOR)
- Never invest more than you can afford to lose

## 📝 License

MIT License - feel free to use for any purpose.

## 🔗 Links

- [Portfolio](https://your-portfolio.com)
- [Binance API Docs](https://binance-docs.github.io/apidocs/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

Made with ❤️ by [Your Name](https://github.com/your-username)
