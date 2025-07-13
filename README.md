# 📈 stock_mf-predict-algo

A powerful, user-friendly Telegram bot for stock and mutual fund analysis, powered by advanced AI models and technical indicators.
This project is designed for retail investors, traders, and finance enthusiasts who want actionable insights, trading signals, and AI-powered recommendations—all from a simple chat interface.

---

### **Key Features**

- **Multi-Model AI Analysis:**
  Integrates with top AI providers (Ollama, Gemini, Hugging Face, Anthropic, OpenAI, DeepSeek, Grok) for stock and mutual fund analysis.
  Users can select their preferred AI model per session, just like in Cursor.

- **Secure API Key Management:**
  API keys are stored in a local `config.py` (never pushed to Git), with a template and `.gitignore` for best security practices.

- **Comprehensive Stock & MF Analysis:**
  - Technical indicators: RSI, MACD, Supertrend, moving averages, and more.
  - AI-powered sentiment, confidence, and trading recommendations.
  - Interactive Plotly charts for visual analysis.
  - News sentiment and layman-friendly advice.

- **Telegram Bot Commands:**
  - `/analyze <symbol>`: Basic technical analysis.
  - `/advanced <symbol>`: Advanced analysis with AI.
  - `/grok <symbol>`: Deep analysis using Grok AI.
  - `/freeai <symbol>`: Use free AI services for analysis.
  - `/ask <symbol> BUY/SELL/HOLD <qty> <price>`: Get advice for specific trades.
  - `/recommend <symbol>`: Detailed trading recommendations (stop loss, targets, holding period, etc.).
  - `/model <model_name>`: Select your preferred AI model/provider.
  - `/help`: List all commands.

- **Per-User Model Selection:**
  Each user can choose their preferred AI model for analysis, with easy switching via `/model`.

- **Mutual Fund Support:**
  Recognizes and analyzes mutual fund symbols, providing tailored advice.

- **Free AI Fallback:**
  Automatically falls back to available free AI providers if one is unavailable or rate-limited.

- **Easy Setup:**
  - One-command setup script (`setup.py`)
  - Clear README and config templates
  - No sensitive data in version control

---

### **Tech Stack**

- **Python 3.8+**
- **Telegram Bot API**
- **Pandas, NumPy, yfinance** (for data)
- **Plotly** (for charts)
- **Multiple AI APIs** (Ollama, Gemini, Hugging Face, Anthropic, OpenAI, DeepSeek, Grok)
- **Pydantic** (for structured AI responses)

---

### **Security**

- All API keys are stored locally in `config.py` (never committed).
- `.gitignore` and `config_template.py` ensure safe sharing and deployment.

---

### **Getting Started**

1. Clone the repo
2. Run `python setup.py`
3. Add your API keys to `config.py`
4. Start the bot: `python backtest.py`
5. Use Telegram commands to analyze stocks and mutual funds

---

### **Use Cases**

- Retail investors seeking AI-powered stock/mutual fund advice
- Traders wanting technical and sentiment analysis in one place
- Developers looking for a secure, extensible AI bot template

---

### **Disclaimer**

This project is for educational purposes only.
It does not constitute financial advice.
Always do your own research and consult a professional before making investment decisions.

---

**Happy Trading! 🚀**