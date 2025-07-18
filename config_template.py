# Configuration Template
# Copy this file to config.py and fill in your non-secret configuration
# All API keys and OAuth client IDs/secrets are now stored encrypted in the local SQLite DB.
# Manage secrets via the admin UI in the Streamlit app sidebar (admin login required).

# Example of non-secret config:

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"  # Default model name

# Other Configuration
YAHOO_FINANCE_ENABLED = True  # Use Yahoo Finance as fallback

# Analysis Configuration
DEFAULT_ANALYSIS_DEPTH = "basic"  # "basic" or "advanced"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Admin credentials (for initial login, can be changed later)
USERNAME = "admin"
PASSWORD = "admin"

# Encryption key for SQLite DB (change this!)
DB_ENCRYPTION_KEY = "mysecretkey"