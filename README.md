# 📈 stock_mf-predict-algo

A powerful, user-friendly Telegram bot for stock and mutual fund analysis, powered by advanced AI models and technical indicators.
This project is designed for retail investors, traders, and finance enthusiasts who want actionable insights, trading signals, and AI-powered recommendations—all from a simple chat interface.

---

## 📝 Changelog (Recent Major Changes)

**July 2024:**
- **Deployment & API:**
  - Added a full-featured Streamlit dashboard (`dashboard.py`) with admin login, secret management, and a built-in REST API (FastAPI, port 8000).
  - REST API now supports unified recommendations, OI chart, and backtest CSV download. JWT and API key authentication supported.
  - PDF export utility for analytics reports.
- **AI & ML Enhancements:**
  - Multi-provider AI integration: Ollama (local), Gemini, HuggingFace, Anthropic, OpenAI, DeepSeek, Grok.
  - Per-user model selection and automatic free AI fallback.
  - Improved news sentiment analysis and layman-friendly advice.
  - Enhanced technical indicators, ML-based predictions, and backtesting (with `backtrader`).
- **Security:**
  - All API keys and secrets are now managed via the Streamlit admin UI and stored encrypted in a local SQLite DB (never in `config.py`).
  - `.gitignore` and `config_template.py` ensure no secrets are committed.
- **Config & Setup:**
  - `config_template.py` now only contains non-secret config. All secrets are managed via the dashboard.
  - `requirements.txt` updated for all new dependencies (see below).
- **Broker Integrations:**
  - Modular broker classes for Zerodha, Alpaca, Breeze, and placeholders for Groww/ICICI.
- **Other Improvements:**
  - Improved error handling, modularity, and code structure across all files.
  - Expanded and clarified documentation throughout the codebase and README.

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

### **Deployment**

- **Streamlit Dashboard:**
  1. Run `streamlit run dashboard.py` to launch the dashboard UI (default port 8501).
  2. Log in as admin to set API keys, secrets, and the REST API key in the sidebar.
  3. The dashboard will also start a REST API (FastAPI) server on port 8000 by default.

- **REST API (Unified Recommendation):**
  - All `/api/unified` endpoints require an `X-API-Key` header for authentication.
  - Set the API key in the dashboard sidebar (admin only).
  - Example usage:
    ```bash
    curl -H "X-API-Key: <your_api_key>" "http://localhost:8000/api/unified?symbol=NIFTY"
    ```
  - For production, run behind a reverse proxy (e.g., nginx) and enable HTTPS.
  - **Advanced:** You can enable JWT authentication for the API by replacing the API key logic with JWT token validation (see FastAPI docs for details).

- **Telegram Bot:**
  - Start the bot with `python backtest.py` (ensure your Telegram bot token is set in the dashboard or config).

- **Environment Variables:**
  - For production, set sensitive values (API keys, DB encryption key) via environment variables or secrets management.

- **Security Best Practices:**
  - Never commit `config.py` or secrets to version control.
  - Use strong, unique API keys or JWT secrets.
  - Always use HTTPS in production.

---

### **Disclaimer**

This project is for educational purposes only.
It does not constitute financial advice.
Always do your own research and consult a professional before making investment decisions.

---
> ** Note:**
> This project demonstrates my ability to build secure, production-ready AI applications for finance, integrating multiple AI providers, advanced analytics, and a modern user interface. I am passionate about leveraging technology to empower users and solve real-world problems.

**Happy Trading! 🚀**

## 👨‍💻 About the Author

Hi, This is Aman Sharma, a software engineer specializing in AI, fintech, and secure application development. This project showcases my skills in:
- End-to-end product development (from backend to UI)
- Secure API and secret management
- Integrating multiple AI/ML providers
- Building user-friendly dashboards and bots
- Writing clean, well-documented, and production-ready code

Let's connect on [LinkedIn](https://www.linkedin.com/in/aman-sharma-53a2a9117/) or [email](rosellete.william@gmail.com)!

## 🛠️ Key Technologies & Skills Demonstrated

- Python 3.8+, Streamlit, FastAPI, Telegram Bot API
- Secure API key management (encrypted, never in VCS)
- Multi-provider AI integration (Ollama, Gemini, OpenAI, etc.)
- Data analysis: Pandas, NumPy, yfinance
- Interactive visualization: Plotly
- REST API design & authentication (API key, JWT)
- Modular, extensible codebase
- Deployment & documentation best practices

## 📸 Screenshots

![Dashboard Screenshot](path/to/dashboard_screenshot.png)
![Telegram Bot in Action](path/to/bot_screenshot.png)

---
