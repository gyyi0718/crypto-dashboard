# 📈 Crypto Paper Trading Dashboard

Real-time cryptocurrency paper trading with persistent storage.

![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

## 🌐 Live Demo

**[👉 View Live Dashboard](https://your-app-name.streamlit.app)**

## ✨ Features

- **Real-time Price**: BTC, ETH, SOL, XRP, DOGE, BNB (Yahoo Finance)
- **Paper Trading**: Long/Short positions with leverage
- **PnL Tracking**: Realized & unrealized profit/loss
- **Persistent Storage**: Trade history saved to Supabase
- **Technical Indicators**: RSI, MACD, EMA
- **AI Signals**: Automated buy/sell recommendations

## 🚀 Setup

### 1. Supabase Setup

1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Go to SQL Editor and run `supabase_setup.sql`
4. Get API keys from Project Settings → API

### 2. Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy this repo
4. Add Secrets:

```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJxxxx..."
```

### 3. Local Development

```bash
# Clone
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard

# Install
pip install -r requirements.txt

# Create .env or .streamlit/secrets.toml
# Add SUPABASE_URL and SUPABASE_KEY

# Run
streamlit run streamlit_app.py
```

## 📁 Files

```
├── streamlit_app.py      # Main app
├── requirements.txt      # Dependencies
├── supabase_setup.sql    # Database schema
└── README.md
```

## ⚠️ Disclaimer

This is **paper trading** for educational purposes only.
Not financial advice. No real money involved.

## 🔗 Links

- Data: [Yahoo Finance](https://finance.yahoo.com/)
- Database: [Supabase](https://supabase.com/)
- Hosting: [Streamlit Cloud](https://streamlit.io/cloud)
