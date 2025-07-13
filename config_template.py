# Configuration Template
# Copy this file to config.py and fill in your actual API keys

# Grok API Configuration
GROK_API_KEY = "your_grok_api_key_here"

# Free AI API Keys
GEMINI_API_KEY = "your_gemini_api_key_here"
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"  # Default model name

# Other Configuration
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key_here"  # If you have one
YAHOO_FINANCE_ENABLED = True  # Use Yahoo Finance as fallback

# Bot Configuration
BOT_TOKEN = "your_telegram_bot_token_here"
CHAT_ID = "your_chat_id_here"  # Your Telegram chat ID

# Analysis Configuration
DEFAULT_ANALYSIS_DEPTH = "basic"  # "basic" or "advanced"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30