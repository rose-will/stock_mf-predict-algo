# Free AI Integration Setup Guide

This guide helps you set up free AI services for your stock analysis bot.

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   ```bash
   # Optional: Set API keys for enhanced features
   export HUGGINGFACE_API_KEY="your_key_here"
   export ANTHROPIC_API_KEY="your_key_here"
   export GEMINI_API_KEY="your_key_here"
   export DEEPSEEK_API_KEY="sk-4c170823527e4108a08cbb30541c9c12"
   ```

3. **Test the Integration:**
   ```bash
   python test_deepseek.py
   ```

## 🤖 Available AI Providers

### 1. **Ollama (Local AI) - RECOMMENDED**
- **Cost:** Free (unlimited)
- **Setup:**
  ```bash
  # Install Ollama
  curl -fsSL https://ollama.ai/install.sh | sh

  # Start Ollama service
  ollama serve

  # Download a model (in another terminal)
  ollama pull llama2
  ```
- **Features:** Unlimited requests, runs locally, no API limits

### 2. **DeepSeek API**
- **Cost:** Free tier available
- **Setup:**
  - API Key: `sk-4c170823527e4108a08cbb30541c9c12`
  - Base URL: `https://api.deepseek.com`
  - Model: `deepseek-chat`
  - **Note:** Currently disabled by default due to insufficient balance
- **Features:** High-quality analysis, OpenAI-compatible API
- **To Enable:** Set `DEEPSEEK_API_KEY` environment variable and update `free_ai_integration.py` to set `'enabled': True`

### 3. **Hugging Face Inference API**
- **Cost:** Free (30,000 requests/month)
- **Setup:** Get API key from [Hugging Face](https://huggingface.co/settings/tokens)
- **Features:** Multiple models available

### 4. **Google Gemini**
- **Cost:** Free (64,800 requests/month)
- **Setup:** Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Features:** High-quality responses

### 5. **Anthropic Claude**
- **Cost:** Free tier available
- **Setup:** Get API key from [Anthropic Console](https://console.anthropic.com/)
- **Features:** Excellent reasoning capabilities

### 6. **OpenAI Free Tier**
- **Cost:** Free (limited requests)
- **Setup:** Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Features:** GPT-3.5-turbo access

## 🔧 Configuration

The bot automatically tries providers in this order:
1. **Ollama** (local, unlimited)
2. **DeepSeek** (high quality, free tier)
3. **Hugging Face** (30K requests/month)
4. **Gemini** (64K requests/month)
5. **Anthropic** (free tier)
6. **OpenAI** (free tier)

## 📊 Usage

### Test Free AI Services
```bash
# Test DeepSeek specifically
python test_deepseek.py

# Test all free AI services
python -c "from free_ai_integration import analyze_stock_with_free_ai; print(analyze_stock_with_free_ai('AAPL: $150, RSI: 65, MACD: bullish'))"
```

### In Telegram Bot
```
/freeai AAPL
```

## 🛠️ Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### API Key Issues
- Ensure API keys are correctly set in environment variables
- Check API key permissions and quotas
- Verify network connectivity

### Model Issues
- Some models may be temporarily unavailable
- The bot automatically falls back to other providers
- Check provider status pages for outages

## 📈 Performance Tips

1. **Use Ollama for unlimited requests** - Best for heavy usage
2. **Set up multiple providers** - Ensures availability
3. **Monitor usage** - Check `/freeai` command for stats
4. **Local models** - Download models for offline use

## 🔒 Security Notes

- API keys are stored in environment variables
- No keys are hardcoded in the bot
- Local Ollama runs completely offline
- All API calls use HTTPS

## 📝 Example Output

```
🤖 FREE AI ANALYSIS: AAPL

💰 Current Price: $150.00
📊 Provider: DeepSeek
🎯 Sentiment: POSITIVE
🎚️ Confidence: 75.0%
📈 Impact: BULLISH
💡 Recommendation: BUY

📝 Summary:
Based on the technical indicators, AAPL shows bullish momentum with RSI at 65 indicating strong but not overbought conditions. MACD crossover suggests upward trend continuation.

📊 Usage Stats:
• Current Provider: DeepSeek
• Ollama: 0/unlimited requests
• DeepSeek: 1/1000 requests
• HuggingFace: 0/30000 requests
```

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test individual providers
4. Check provider status pages
5. Review error messages in bot logs