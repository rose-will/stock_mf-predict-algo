# Stock Analysis Bot with AI Integration

A comprehensive Telegram bot for stock analysis using multiple AI services including Grok, Gemini, Hugging Face, Ollama, and more.

## Features

- 📊 **Technical Analysis**: RSI, MACD, Supertrend, and more indicators
- 🤖 **AI Integration**: Multiple free AI services for sentiment analysis
- 📈 **Interactive Charts**: Plotly-based interactive stock charts
- 💡 **Trading Recommendations**: Buy/Sell/Hold advice with confidence scores
- 🔄 **Free AI Fallback**: Automatic fallback between multiple AI providers
- 📱 **Telegram Bot**: Easy-to-use commands and interactive buttons

## Commands

- `/start` - Welcome message and help
- `/analyze <symbol>` - Basic stock analysis
- `/advanced <symbol>` - Advanced analysis with AI
- `/grok <symbol>` - Deep analysis using Grok AI
- `/freeai <symbol>` - Analysis using free AI services
- `/ask <symbol> BUY/SELL/HOLD <quantity> <price>` - Specific trading advice
- `/recommend <symbol>` - Detailed trading recommendations
- `/help` - Show all available commands

## New: Per-User AI Model Selection

You can now choose which AI model/provider to use for your analysis, just like in Cursor!

### /model Command

- `/model` — Shows your current model and all available options
- `/model <model_name>` — Sets your preferred model (e.g., `/model gemini`)

Your choice is remembered for your session. All analysis commands (`/freeai`, `/ask`, `/recommend`, etc.) will use your selected model if set, otherwise the default fallback order is used.

**Available models:**
- ollama
- gemini
- huggingface
- anthropic
- openai_free
- deepseek
- grok

**Example:**
```
/model gemini
/freeai TCS.NS
```
This will analyze TCS.NS using Gemini for your user session.

If you want to switch back to automatic fallback, just use `/model` and do not select a model.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

**IMPORTANT**: Never commit your actual API keys to Git!

1. Copy the template file:
   ```bash
   cp config_template.py config.py
   ```

2. Edit `config.py` and add your API keys:
   ```python
   # Grok API Configuration
   GROK_API_KEY = "your_actual_grok_api_key"

   # Free AI API Keys
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   HUGGINGFACE_API_KEY = "your_actual_huggingface_api_key"
   ANTHROPIC_API_KEY = "your_actual_anthropic_api_key"
   OPENAI_API_KEY = "your_actual_openai_api_key"

   # Bot Configuration
   BOT_TOKEN = "your_actual_telegram_bot_token"
   CHAT_ID = "your_actual_chat_id"
   ```

### 3. Get API Keys

#### Telegram Bot Token
1. Message @BotFather on Telegram
2. Create a new bot with `/newbot`
3. Copy the token to `BOT_TOKEN`

#### Free AI Services

**Google Gemini** (Recommended - 15 requests/minute):
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create API key
- Add to `GEMINI_API_KEY`

**Hugging Face** (30,000 requests/month):
- Visit [Hugging Face](https://huggingface.co/settings/tokens)
- Create access token
- Add to `HUGGINGFACE_API_KEY`

**Anthropic Claude** (5 requests/minute):
- Visit [Anthropic Console](https://console.anthropic.com/)
- Create API key
- Add to `ANTHROPIC_API_KEY`

**OpenAI** (Limited free tier):
- Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- Create API key
- Add to `OPENAI_API_KEY`

**Ollama** (Local - Unlimited):
- Install [Ollama](https://ollama.ai/)
- Run `ollama pull llama3.2`
- Start with `ollama serve`

### 4. Run the Bot

```bash
python backtest.py
```

## Security

- ✅ `config.py` is in `.gitignore` - never committed to Git
- ✅ `config_template.py` shows structure without real keys
- ✅ Environment variables supported as fallback
- ✅ API keys are loaded securely

## File Structure

```
├── backtest.py              # Main bot file
├── free_ai_integration.py   # Free AI services integration
├── config.py               # API keys (not in Git)
├── config_template.py      # Template for config
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Supported Stock Symbols

- **US Stocks**: AAPL, GOOGL, MSFT, etc.
- **Indian Stocks**: TCS.NS, RELIANCE.NS, etc.
- **Mutual Funds**: Basic support for fund symbols

## AI Provider Priority

1. **Ollama** (Local) - Unlimited requests
2. **Gemini** - 15 requests/minute
3. **Hugging Face** - 30,000 requests/month
4. **Anthropic** - 5 requests/minute
5. **OpenAI** - Limited free tier
6. **DeepSeek** - Disabled (insufficient balance)

## Troubleshooting

### Bot Not Responding
- Check `BOT_TOKEN` in config.py
- Ensure bot is added to your chat
- Check `CHAT_ID` is correct

### AI Analysis Failing
- Verify API keys are correct
- Check internet connection
- Try different AI providers

### Ollama Not Working
- Install Ollama: https://ollama.ai/
- Run `ollama serve`
- Pull a model: `ollama pull llama3.2`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Use at your own risk for trading decisions.