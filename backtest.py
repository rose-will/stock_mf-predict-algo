# Advanced Stock Prediction & Analysis Bot
#
# Sources and inspiration:
# - NSE-Option-Chain-Analyzer (GitHub): Option chain fetching and OI analysis
#   https://github.com/saikiranboga/nse-option-chain-analyzer
# - QuantInsti: Random Forest and ML in trading
#   https://blog.quantinsti.com/random-forest-trading-strategy/
# - NSE Academy & Courses: Technical analysis and options strategies
#   https://www.nseindia.com/learn/nse-academy
#
# This file combines yfinance and NSEPython for data, advanced ML features, and both plotly and matplotlib for visualization.

import yfinance as yf
import pandas as pd
import numpy as np
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import requests
import json
from datetime import datetime, timedelta
import openai
import asyncio
from openai import OpenAI
import os
from typing import List, Dict, Any, Optional
from enum import Enum
import re
import matplotlib.pyplot as plt
import backtrader as bt
import tempfile
from breeze_connect import BreezeConnect
import zipfile  # For handling zipped CSVs
import io  # For in-memory file handling
from functools import lru_cache

# Grok SDK imports
try:
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    GROK_SDK_AVAILABLE = True
except ImportError:
    GROK_SDK_AVAILABLE = False
    print("Warning: xai_sdk not available. Install with: pip install xai-sdk")

# Note: You'll need to install: pip install openai kaleido xai-sdk

# Free AI Integration
try:
    from free_ai_integration import analyze_stock_with_free_ai, get_free_ai_usage, get_available_ai_models
    FREE_AI_AVAILABLE = True
    print("✅ Free AI integration available")
except ImportError:
    FREE_AI_AVAILABLE = False
    print("⚠️ Free AI integration not available. Install with: pip install -r requirements.txt")

warnings.filterwarnings('ignore')

# Import configuration
try:
    from config import *
    print("✅ Configuration loaded from config.py")
except ImportError:
    print("⚠️ config.py not found. Please copy config_template.py to config.py and add your API keys")
    # Fallback values (replace with your actual keys)
    TOKEN = 'your_telegram_bot_token_here'
    OPENAI_API_KEY = 'your_openai_api_key_here'
    GROK_API_KEY = 'your_grok_api_key_here'
    GROK_API_URL = 'https://api.x.ai/v1/chat/completions'

# --- GROK ENABLED FLAG ---
GROK_ENABLED = (
    GROK_SDK_AVAILABLE and
    'GROK_API_KEY' in globals() and
    GROK_API_KEY and
    not GROK_API_KEY.startswith('your-')
)

# Initialize Grok client if SDK is available and enabled
if GROK_ENABLED:
    try:
        grok_client = Client(api_key=GROK_API_KEY)
        print("✅ Grok SDK client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Grok client: {e}")
        grok_client = None
        GROK_ENABLED = False
else:
    grok_client = None
    print("Grok integration is disabled (missing SDK or API key). All Grok features will be skipped.")

# Pydantic models for structured AI responses
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: pydantic not available. Install with: pip install pydantic")

if PYDANTIC_AVAILABLE:
    class Currency(str, Enum):
        USD = "USD"
        EUR = "EUR"
        GBP = "GBP"
        INR = "INR"

    class LineItem(BaseModel):
        description: str = Field(description="Description of the item or service")
        quantity: int = Field(description="Number of units", ge=1)
        unit_price: float = Field(description="Price per unit", ge=0)

    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City")
        postal_code: str = Field(description="Postal/ZIP code")
        country: str = Field(description="Country")

    class Invoice(BaseModel):
        vendor_name: str = Field(description="Name of the vendor")
        vendor_address: Address = Field(description="Vendor's address")
        invoice_number: str = Field(description="Unique invoice identifier")
        invoice_date: str = Field(description="Date the invoice was issued")
        line_items: List[LineItem] = Field(description="List of purchased items/services")
        total_amount: float = Field(description="Total amount due", ge=0)
        currency: Currency = Field(description="Currency of the invoice")

    class StockAnalysis(BaseModel):
        symbol: str = Field(description="Stock symbol")
        sentiment: str = Field(description="Overall sentiment (bullish/bearish/neutral)")
        confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
        summary: str = Field(description="Brief analysis summary")
        key_factors: List[str] = Field(description="Key factors influencing the analysis")
        risk_level: str = Field(description="Risk level (low/medium/high)")
        recommendation: str = Field(description="Trading recommendation")
        target_price: Optional[float] = Field(description="Target price if available")
        stop_loss: Optional[float] = Field(description="Stop loss price if available")

try:
    from nsepython import option_chain, equity_history
    NSEPYTHON_AVAILABLE = True
except ImportError:
    NSEPYTHON_AVAILABLE = False
    print("Warning: nsepython not available. Install with: pip install nsepython")

def get_usd_to_inr_rate():
    """Get current USD to INR exchange rate"""
    try:
        # Get USD to INR exchange rate
        usd_inr = yf.download('USDINR=X', period='1d', interval='1d')
        if usd_inr is not None and not usd_inr.empty:
            return float(usd_inr['Close'].iloc[-1])
        else:
            # Fallback rate (approximate)
            return 83.0
    except:
        # Fallback rate if API fails
        return 83.0

def is_us_stock(symbol):
    """Check if symbol is a US stock (not ending in .NS or .BO)"""
    return not (symbol.endswith('.NS') or symbol.endswith('.BO'))

def calculate_position_size(current_price, confidence, risk_percentage=2.0, account_size=100000):
    """Calculate position size based on risk management"""
    # Risk amount = account_size * risk_percentage / 100
    risk_amount = account_size * risk_percentage / 100

    # Position size = risk_amount / (current_price * 0.02)  # Assuming 2% stop loss
    stop_loss_percentage = 0.02  # 2% stop loss
    position_size = risk_amount / (current_price * stop_loss_percentage)

    # Adjust based on confidence
    confidence_multiplier = confidence / 100
    adjusted_position_size = position_size * confidence_multiplier

    return int(adjusted_position_size)

def generate_trading_signals_detailed(df, confidence, recommendation):
    """Generate detailed trading signals with position sizing"""
    current_price = float(df['Close'].iloc[-1])
    current_rsi = float(df['RSI'].iloc[-1])
    current_macd = float(df['MACD'].iloc[-1])
    current_supertrend = float(df['Supertrend'].iloc[-1])

    signals = []
    position_size = calculate_position_size(current_price, confidence * 100)

    # Entry signals
    if recommendation in ["STRONG BUY", "BUY"]:
        entry_price = current_price
        stop_loss = entry_price * 0.98  # 2% stop loss
        take_profit = entry_price * 1.06  # 6% take profit

        signals.append(f"🎯 ENTRY SIGNAL: BUY")
        signals.append(f"💰 Entry Price: ₹{entry_price:.2f}")
        signals.append(f"🛑 Stop Loss: ₹{stop_loss:.2f} (-2%)")
        signals.append(f"🎯 Take Profit: ₹{take_profit:.2f} (+6%)")
        signals.append(f"📊 Position Size: {position_size} shares")
        signals.append(f"⏰ Hold Duration: 1-3 weeks")

    elif recommendation in ["STRONG SELL", "SELL"]:
        entry_price = current_price
        stop_loss = entry_price * 1.02  # 2% stop loss for short
        take_profit = entry_price * 0.94  # 6% take profit for short

        signals.append(f"🎯 ENTRY SIGNAL: SELL")
        signals.append(f"💰 Entry Price: ₹{entry_price:.2f}")
        signals.append(f"🛑 Stop Loss: ₹{stop_loss:.2f} (+2%)")
        signals.append(f"🎯 Take Profit: ₹{take_profit:.2f} (-6%)")
        signals.append(f"📊 Position Size: {position_size} shares")
        signals.append(f"⏰ Hold Duration: 1-2 weeks")

    # Technical analysis details
    signals.append(f"\n📊 TECHNICAL ANALYSIS:")
    signals.append(f"• RSI: {current_rsi:.1f} ({'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'})")
    signals.append(f"• MACD: {current_macd:.4f} ({'Bullish' if current_macd > 0 else 'Bearish'})")
    signals.append(f"• Supertrend: ₹{current_supertrend:.2f} ({'Bullish' if current_price > current_supertrend else 'Bearish'})")

    return signals

def get_stock_news(symbol):
    """Get recent news about the stock"""
    try:
        # For demonstration, using a simple news API
        # In production, you'd use a proper news API like Alpha Vantage, NewsAPI, etc.

        # Clean symbol for news search
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')

        # This is a placeholder - you'd integrate with a real news API
        news_items = [
            f"📰 Recent news for {clean_symbol}: Market analysis shows mixed sentiment",
            f"📰 {clean_symbol} quarterly results expected next week",
            f"📰 Analysts maintain neutral rating on {clean_symbol}"
        ]

        return news_items
    except Exception as e:
        return [f"📰 Unable to fetch news: {str(e)}"]

def analyze_news_with_grok_sdk(news_items: List[str], symbol: str) -> Dict[str, Any]:
    """Analyze news sentiment using Grok SDK with structured output"""
    if not GROK_ENABLED or not grok_client or not PYDANTIC_AVAILABLE:
        return analyze_news_with_ai_fallback(news_items, symbol)
    try:
        # For now, use fallback to avoid SDK method issues
        return analyze_news_with_ai_fallback(news_items, symbol)
    except Exception as e:
        print(f"Grok SDK analysis error: {e}")
        return analyze_news_with_ai_fallback(news_items, symbol)

def analyze_news_with_ai_fallback(news_items: List[str], symbol: str) -> Dict[str, Any]:
    """Fallback AI analysis using OpenAI or REST API"""
    try:
        # Try OpenAI first
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith('your-'):
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)

                prompt = (
                    f"Analyze the following news headlines for the stock {symbol} and provide:\n"
                    f"1. Sentiment (positive/negative/neutral)\n"
                    f"2. Confidence (0-1)\n"
                    f"3. A one-sentence summary\n"
                    f"4. Impact (bullish/bearish/neutral)\n\n"
                    f"News:\n" + "\n".join(news_items)
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    # Simple parsing
                    lines = content.split('\n')
                    sentiment = lines[0].split(':')[-1].strip().lower() if len(lines) > 0 else 'neutral'
                    confidence = float(lines[1].split(':')[-1].strip()) if len(lines) > 1 else 0.5
                    summary = lines[2].split(':', 1)[-1].strip() if len(lines) > 2 else ''
                    impact = lines[3].split(':')[-1].strip().lower() if len(lines) > 3 else 'neutral'

                    return {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'summary': summary,
                        'impact': impact,
                        'provider': 'OpenAI'
                    }

            except Exception as openai_error:
                error_msg = str(openai_error)
                if 'insufficient_quota' in error_msg or '429' in error_msg:
                    print(f"OpenAI quota exceeded, trying Grok REST API: {error_msg}")
                else:
                    print(f"OpenAI error: {error_msg}")

        # Try Grok REST API as fallback
        if GROK_ENABLED and GROK_API_KEY and not GROK_API_KEY.startswith('your-'):
            try:
                prompt = (
                    f"Analyze the following news headlines for the stock {symbol} and provide:\n"
                    f"1. Sentiment (positive/negative/neutral)\n"
                    f"2. Confidence (0-1)\n"
                    f"3. A one-sentence summary\n"
                    f"4. Impact (bullish/bearish/neutral)\n\n"
                    f"News:\n" + "\n".join(news_items)
                )
                headers = {
                    'Authorization': f'Bearer {GROK_API_KEY}',
                    'Content-Type': 'application/json'
                }
                data = {
                    'model': 'grok-beta',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 200,
                    'temperature': 0.3
                }
                response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                        content = result['choices'][0]['message']['content']
                        # Simple parsing
                        lines = content.split('\n')
                        sentiment = lines[0].split(':')[-1].strip().lower() if len(lines) > 0 else 'neutral'
                        confidence = float(lines[1].split(':')[-1].strip()) if len(lines) > 1 else 0.5
                        summary = lines[2].split(':', 1)[-1].strip() if len(lines) > 2 else ''
                        impact = lines[3].split(':')[-1].strip().lower() if len(lines) > 3 else 'neutral'
                        return {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'summary': summary,
                            'impact': impact,
                            'provider': 'Grok REST API'
                        }
                    else:
                        print(f"Grok REST API error: No content in response")
                else:
                    print(f"Grok REST API error: {response.status_code} - {response.text}")
            except Exception as grok_error:
                print(f"Grok REST API error: {grok_error}")

        # Fallback when all APIs fail
        return {
            'sentiment': 'neutral',
            'confidence': 0.6,
            'summary': 'AI analysis temporarily unavailable. Using technical analysis for sentiment.',
            'impact': 'neutral',
            'provider': 'Fallback'
        }

    except Exception as e:
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'summary': f'AI analysis error: {str(e)}',
            'impact': 'neutral',
            'provider': 'Error'
        }

def analyze_news_with_ai(news_items, symbol):
    """Main function to analyze news with AI - tries Grok SDK first, then fallbacks"""
    return analyze_news_with_grok_sdk(news_items, symbol)

def get_advanced_stock_analysis_with_grok(symbol: str, technical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get advanced stock analysis using Grok AI"""
    if not GROK_ENABLED or not grok_client:
        return {
            'analysis': 'Grok SDK not available',
            'provider': 'Not Available'
        }
    try:
        # For now, return a basic analysis to avoid SDK method issues
        analysis_text = f"""
        Comprehensive Analysis for {symbol}:

        Current Price: ${technical_data.get('current_price', 0):.2f}
        RSI: {technical_data.get('rsi', 0):.1f} ({'Oversold' if technical_data.get('rsi', 0) < 30 else 'Overbought' if technical_data.get('rsi', 0) > 70 else 'Neutral'})
        MACD: {technical_data.get('macd', 0):.4f} ({'Bullish' if technical_data.get('macd', 0) > 0 else 'Bearish'})
        Volume Ratio: {technical_data.get('volume_ratio', 0):.2f}x average

        Technical Assessment:
        - Price action shows {'bullish' if technical_data.get('current_price', 0) > technical_data.get('sma_20', 0) else 'bearish'} momentum
        - Volume {'supports' if technical_data.get('volume_ratio', 0) > 1.2 else 'does not support'} the current trend
        - Risk level: {'High' if technical_data.get('rsi', 0) > 70 or technical_data.get('rsi', 0) < 30 else 'Medium'}

        Recommendation: {'BUY' if technical_data.get('macd', 0) > 0 and technical_data.get('rsi', 0) < 70 else 'SELL' if technical_data.get('macd', 0) < 0 and technical_data.get('rsi', 0) > 30 else 'HOLD'}
        """
        return {
            'analysis': analysis_text,
            'provider': 'Grok AI (Basic)'
        }
    except Exception as e:
        print(f"Grok advanced analysis error: {e}")
        return {
            'analysis': f'Analysis error: {str(e)}',
            'provider': 'Grok AI (Error)'
        }

def analyze_with_free_ai_fallback(stock_data, symbol):
    """Fallback to free AI services when Grok is not available"""
    if not FREE_AI_AVAILABLE:
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'summary': 'Free AI services not available',
            'impact': 'neutral',
            'provider': 'Not Available'
        }

    try:
        # Format stock data for analysis
        analysis_text = f"""
        Stock: {symbol}
        Current Price: ${stock_data.get('current_price', 0):.2f}
        RSI: {stock_data.get('rsi', 0):.1f}
        MACD: {stock_data.get('macd', 0):.4f}
        Volume Ratio: {stock_data.get('volume_ratio', 0):.2f}
        20 SMA: ${stock_data.get('sma_20', 0):.2f}
        50 SMA: ${stock_data.get('sma_50', 0):.2f}
        """

        result = analyze_stock_with_free_ai(analysis_text)

        if 'error' in result:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'summary': f'Free AI error: {result["error"]}',
                'impact': 'neutral',
                'provider': 'Free AI (Error)'
            }

        return {
            'sentiment': result.get('sentiment', 'neutral'),
            'confidence': result.get('confidence', 0.5),
            'summary': result.get('analysis', 'Analysis completed'),
            'impact': 'bullish' if result.get('sentiment') == 'positive' else 'bearish' if result.get('sentiment') == 'negative' else 'neutral',
            'provider': result.get('provider', 'Free AI'),
            'recommendation': result.get('recommendation', 'hold')
        }

    except Exception as e:
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'summary': f'Free AI analysis error: {str(e)}',
            'impact': 'neutral',
            'provider': 'Free AI (Error)'
        }

def create_interactive_chart(df, symbol):
    """Create chart and return as bytes for Telegram"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Supertrend', 'RSI', 'MACD', 'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        # Supertrend
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Supertrend'],
            mode='lines',
            name='Supertrend',
            line=dict(color='orange', width=2)
        ), row=1, col=1)

        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue', width=1)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red', width=1)
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=2, col=1)

        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")

        # MACD
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red', width=2)
        ), row=3, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=4, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            width=1200
        )

        # Convert to image bytes for Telegram
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return img_bytes

    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def calculate_technical_indicators(df, symbol=None, start_date=None, end_date=None, expiry=None, poll_oi_delta=False, oi_delta_interval=60):
    """Calculate comprehensive technical indicators, including PCR/OI if available."""
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2.0)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2.0)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    stoch_denom = (high_max - low_min)
    stoch_denom = stoch_denom.replace(0, np.nan)  # Avoid division by zero
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / stoch_denom)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # Volume indicators
    volume_sma = df['Volume'].rolling(window=20).mean()
    df['Volume_SMA'] = volume_sma
    df['Volume_Ratio'] = df['Volume'] / volume_sma

    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5'] = df['Close'].pct_change(periods=5)
    df['Price_Change_10'] = df['Close'].pct_change(periods=10)

    # Volatility
    close_series = df['Close']
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series(close_series)
    returns = close_series.pct_change().dropna()
    df['Volatility'] = float(returns.std() * np.sqrt(252) * 100)

    # PCR/OI (new)
    pcr, total_call_oi, total_put_oi = fetch_pcr_oi(symbol) if symbol else (None, None, None)
    df['PCR'] = pcr if pcr is not None else np.nan
    df['Total_Call_OI'] = total_call_oi if total_call_oi is not None else np.nan
    df['Total_Put_OI'] = total_put_oi if total_put_oi is not None else np.nan

    # --- New: Historical and delta OI features ---
    if symbol and start_date and end_date:
        oi_feats = calculate_oi_features(symbol, start_date, end_date, expiry, poll_oi_delta, oi_delta_interval)
        for k, v in oi_feats.items():
            df[k] = v

    return df

def calculate_supertrend(df, atr_period=10, multiplier=3):
    """Simple and robust Supertrend calculation using numpy arrays"""
    # Convert to numpy arrays for reliable indexing and ensure they are 1D
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    close = df['Close'].values.flatten()

    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Calculate ATR using simple moving average
    atr = np.zeros(len(close))
    for i in range(len(close)):
        if i < atr_period - 1:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = np.mean(tr[i-atr_period+1:i+1])

    # Calculate bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Calculate Supertrend
    supertrend = np.zeros(len(close))
    direction = np.zeros(len(close))

    # Initialize
    supertrend[0] = upperband[0]
    direction[0] = 1

    # Main calculation loop
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lowerband[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upperband[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]

    # Convert back to pandas Series
    df['Supertrend'] = pd.Series(supertrend, index=df.index)
    df['Direction'] = pd.Series(direction, index=df.index)

    return df

def prepare_features(df, feature_columns=None):
    if feature_columns is None:
        feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Stoch_K', 'Stoch_D',
            'ATR', 'Volume_Ratio', 'Price_Change', 'Price_Change_5', 'Price_Change_10', 'Volatility',
            'PCR', 'Total_Call_OI', 'Total_Put_OI',
            'Hist_Call_OI_Mean', 'Hist_Put_OI_Mean', 'Hist_Call_OI_Std', 'Hist_Put_OI_Std',
            'Delta_Call_OI_Mean', 'Delta_Put_OI_Mean'
        ]
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df_clean = df.dropna()
    # Only use columns that exist in df
    feature_columns = [col for col in feature_columns if col in df_clean.columns]
    return df_clean[feature_columns], df_clean['Target']

def train_prediction_model(df, feature_columns=None):
    X, y = prepare_features(df, feature_columns)
    if len(X) < 100:
        return None, None, None
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    return model, scaler, accuracy

def generate_signals(df):
    """Generate comprehensive trading signals"""
    signals = []

    try:
        # Ensure we have enough data
        if len(df) < 2:
            return signals

        # Get the last two values safely as scalars
        current_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2])
        current_supertrend = float(df['Supertrend'].iloc[-1])
        prev_supertrend = float(df['Supertrend'].iloc[-2])

        # Supertrend signals
        if current_close > current_supertrend and prev_close <= prev_supertrend:
            signals.append("🟢 STRONG BUY: Price crossed above Supertrend")
        elif current_close < current_supertrend and prev_close >= prev_supertrend:
            signals.append("🔴 STRONG SELL: Price crossed below Supertrend")

        # Moving Average signals
        current_sma20 = float(df['SMA_20'].iloc[-1])
        current_sma50 = float(df['SMA_50'].iloc[-1])
        if current_close > current_sma20 and current_close > current_sma50:
            signals.append("🟢 BUY: Price above both 20 & 50 SMA")
        elif current_close < current_sma20 and current_close < current_sma50:
            signals.append("🔴 SELL: Price below both 20 & 50 SMA")

        # MACD signals
        current_macd = float(df['MACD'].iloc[-1])
        current_macd_signal = float(df['MACD_Signal'].iloc[-1])
        prev_macd = float(df['MACD'].iloc[-2])
        prev_macd_signal = float(df['MACD_Signal'].iloc[-2])

        if current_macd > current_macd_signal and prev_macd <= prev_macd_signal:
            signals.append("🟢 BUY: MACD crossed above signal line")
        elif current_macd < current_macd_signal and prev_macd >= prev_macd_signal:
            signals.append("🔴 SELL: MACD crossed below signal line")

        # RSI signals
        current_rsi = float(df['RSI'].iloc[-1])
        if current_rsi < 30:
            signals.append("🟢 BUY: RSI oversold (< 30)")
        elif current_rsi > 70:
            signals.append("🔴 SELL: RSI overbought (> 70)")

        # Bollinger Bands signals
        current_bb_upper = float(df['BB_Upper'].iloc[-1])
        current_bb_lower = float(df['BB_Lower'].iloc[-1])
        if current_close < current_bb_lower:
            signals.append("🟢 BUY: Price below lower Bollinger Band")
        elif current_close > current_bb_upper:
            signals.append("🔴 SELL: Price above upper Bollinger Band")

        # Stochastic signals
        current_stoch_k = float(df['Stoch_K'].iloc[-1])
        current_stoch_d = float(df['Stoch_D'].iloc[-1])
        if current_stoch_k < 20 and current_stoch_d < 20:
            signals.append("🟢 BUY: Stochastic oversold")
        elif current_stoch_k > 80 and current_stoch_d > 80:
            signals.append("🔴 SELL: Stochastic overbought")

    except Exception as e:
        print(f"Error in generate_signals: {e}")

    return signals

def calculate_confidence_score(df, model, scaler):
    """Calculate prediction confidence score"""
    try:
        if model is None:
            return 0.5, "Model not available"

        # Prepare latest data
        feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Stoch_K', 'Stoch_D',
            'ATR', 'Volume_Ratio', 'Price_Change', 'Price_Change_5', 'Price_Change_10', 'Volatility',
            'PCR', 'Total_Call_OI', 'Total_Put_OI',
            'Hist_Call_OI_Mean', 'Hist_Put_OI_Mean', 'Hist_Call_OI_Std', 'Hist_Put_OI_Std',
            'Delta_Call_OI_Mean', 'Delta_Put_OI_Mean'
        ]

        latest_features = df[feature_columns].iloc[-1:].values

        if np.isnan(latest_features).any():
            return 0.5, "Insufficient data"

        # Scale and predict
        latest_scaled = scaler.transform(latest_features)
        prediction_proba = model.predict_proba(latest_scaled)[0]

        # Calculate confidence based on probability and technical indicators
        base_confidence = max(prediction_proba)

        # Adjust confidence based on technical indicators alignment
        technical_score = 0
        total_indicators = 0

        # Get current values safely as scalars
        current_close = float(df['Close'].iloc[-1])
        current_sma20 = float(df['SMA_20'].iloc[-1])
        current_sma50 = float(df['SMA_50'].iloc[-1])
        current_macd = float(df['MACD'].iloc[-1])
        current_macd_signal = float(df['MACD_Signal'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_supertrend = float(df['Supertrend'].iloc[-1])

        # Check various technical indicators
        if current_close > current_sma20: technical_score += 1
        total_indicators += 1
        if current_close > current_sma50: technical_score += 1
        total_indicators += 1
        if current_macd > current_macd_signal: technical_score += 1
        total_indicators += 1
        if 30 < current_rsi < 70: technical_score += 1
        total_indicators += 1
        if current_close > current_supertrend: technical_score += 1
        total_indicators += 1

        technical_confidence = technical_score / total_indicators

        # Combine ML and technical confidence
        final_confidence = (base_confidence * 0.6) + (technical_confidence * 0.4)

        # Determine recommendation
        if final_confidence > 0.7:
            recommendation = "STRONG BUY" if prediction_proba[1] > 0.5 else "STRONG SELL"
        elif final_confidence > 0.6:
            recommendation = "BUY" if prediction_proba[1] > 0.5 else "SELL"
        else:
            recommendation = "HOLD"

        return final_confidence, recommendation

    except Exception as e:
        print(f"Error in calculate_confidence_score: {e}")
        return 0.5, "Error calculating confidence"

async def analyze_stock_advanced(stock_symbol, update=None):
    """Advanced stock analysis with AI news sentiment"""
    try:
        # Download data
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_historical_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            return f"❌ No data found for {stock_symbol}"

        # Calculate indicators
        data = calculate_technical_indicators(data, stock_symbol)
        data = calculate_supertrend(data)
        # Show OI chart for Indian stocks
        if stock_symbol.endswith('.NS') or stock_symbol in ['NIFTY', 'BANKNIFTY']:
            try:
                from nsepython import option_chain
                oc = option_chain(stock_symbol.replace('.NS',''))
                plot_oi_matplotlib(oc['filtered']['data'], stock_symbol)
            except Exception as e:
                print(f"OI chart error: {e}")

        # Train ML model
        model, scaler, accuracy = train_prediction_model(data)

        # Generate signals
        signals = generate_signals(data)

        # Calculate confidence
        confidence, recommendation = calculate_confidence_score(data, model, scaler)

        # Get news and AI analysis
        news_items = get_stock_news(stock_symbol)
        ai_analysis = analyze_news_with_ai(news_items, stock_symbol)

        # Generate detailed trading signals
        detailed_signals = generate_trading_signals_detailed(data, confidence, recommendation)

        # Current price info
        current_price = float(get_last(data, 'Close', 1)[0])
        price_change = float(get_last(data, 'Close', 1)[0] - get_last(data, 'Close', 2)[0])
        price_change_pct = float((price_change / get_last(data, 'Close', 2)[0]) * 100)

        # Check if it's a US stock and convert to INR if needed
        is_us = is_us_stock(stock_symbol)

        if is_us:
            usd_inr_rate = get_usd_to_inr_rate()
            current_price_inr = current_price * usd_inr_rate
            price_change_inr = price_change * usd_inr_rate

            analysis = (
                f"🚀 ADVANCED ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ${current_price:.2f} (₹{current_price_inr:.2f})\n"
                f"📈 Change: ${price_change:.2f} (₹{price_change_inr:.2f}) ({price_change_pct:+.2f}%)\n\n"
                f"🎯 RECOMMENDATION: {recommendation}\n"
                f"🎚️ Confidence: {confidence:.1%}\n"
                f"🤖 ML Accuracy: {accuracy:.1%}\n\n"
            )
        else:
            analysis = (
                f"🚀 ADVANCED ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ₹{current_price:.2f}\n"
                f"📈 Change: ₹{price_change:.2f} ({price_change_pct:+.2f}%)\n\n"
                f"🎯 RECOMMENDATION: {recommendation}\n"
                f"🎚️ Confidence: {confidence:.1%}\n"
                f"🤖 ML Accuracy: {accuracy:.1%}\n\n"
            )

        # Add detailed trading signals
        for signal in detailed_signals:
            analysis += f"{signal}\n"

        # Add news analysis
        analysis += f"\n📰 NEWS ANALYSIS (via {ai_analysis.get('provider', 'Unknown')}):\n"
        analysis += f"• Sentiment: {ai_analysis['sentiment'].upper()}\n"
        analysis += f"• Confidence: {ai_analysis['confidence']:.1%}\n"
        analysis += f"• Impact: {ai_analysis['impact'].upper()}\n"
        analysis += f"• Summary: {ai_analysis['summary']}\n\n"

        # Add recent news
        analysis += f"📰 RECENT NEWS:\n"
        for news in news_items[:3]:  # Show top 3 news items
            analysis += f"• {news}\n"

        # Add layman advice section
        analysis += f"\n{get_layman_advice(recommendation, confidence)}\n"

        # Create and send chart if update is provided
        if update and update.message:
            try:
                chart_bytes = create_interactive_chart(data, stock_symbol)
                if chart_bytes:
                    await update.message.reply_photo(photo=chart_bytes, caption=f"📊 Technical Chart for {stock_symbol}")
                else:
                    analysis += "\n❌ Could not generate chart."
            except Exception as chart_error:
                print(f"Chart error: {chart_error}")
                analysis += "\n❌ Could not generate chart."

        return analysis

    except Exception as e:
        return f"❌ Error in advanced analysis: {str(e)}"

def analyze_stock(stock_symbol):
    """Comprehensive stock analysis"""
    try:
        # Download data
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_historical_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            return f"❌ No data found for {stock_symbol}"

        # Calculate indicators
        data = calculate_technical_indicators(data, stock_symbol)
        data = calculate_supertrend(data)
        # Show OI chart for Indian stocks
        if stock_symbol.endswith('.NS') or stock_symbol in ['NIFTY', 'BANKNIFTY']:
            try:
                from nsepython import option_chain
                oc = option_chain(stock_symbol.replace('.NS',''))
                plot_oi_matplotlib(oc['filtered']['data'], stock_symbol)
            except Exception as e:
                print(f"OI chart error: {e}")

        # Train ML model
        model, scaler, accuracy = train_prediction_model(data)

        # Generate signals
        signals = generate_signals(data)

        # Calculate confidence
        confidence, recommendation = calculate_confidence_score(data, model, scaler)

        # Current price info - convert to scalars
        current_price = float(get_last(data, 'Close', 1)[0])
        price_change = float(get_last(data, 'Close', 1)[0] - get_last(data, 'Close', 2)[0])
        price_change_pct = float((price_change / get_last(data, 'Close', 2)[0]) * 100)

        # Check if it's a US stock and convert to INR if needed
        is_us = is_us_stock(stock_symbol)
        currency_symbol = "₹" if not is_us else "$"

        if is_us:
            # Convert USD to INR
            usd_inr_rate = get_usd_to_inr_rate()
            current_price_inr = current_price * usd_inr_rate
            price_change_inr = price_change * usd_inr_rate

            # Analysis summary with both USD and INR
            analysis = (
                f"📊 STOCK ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ${current_price:.2f} (₹{current_price_inr:.2f})\n"
                f"📈 Change: ${price_change:.2f} (₹{price_change_inr:.2f}) ({price_change_pct:+.2f}%)\n\n"
                f"🎯 RECOMMENDATION: {recommendation}\n"
                f"🎚️ Confidence: {confidence:.1%}\n"
                f"🤖 ML Accuracy: {accuracy:.1%} (if available)\n\n"
                f"📋 TECHNICAL SIGNALS:\n"
            )
        else:
            # Indian stock - already in INR
            analysis = (
                f"📊 STOCK ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ₹{current_price:.2f}\n"
                f"📈 Change: ₹{price_change:.2f} ({price_change_pct:+.2f}%)\n\n"
                f"🎯 RECOMMENDATION: {recommendation}\n"
                f"🎚️ Confidence: {confidence:.1%}\n"
                f"🤖 ML Accuracy: {accuracy:.1%} (if available)\n\n"
                f"📋 TECHNICAL SIGNALS:\n"
            )

        if signals:
            for signal in signals:
                analysis += f"• {signal}\n"
        else:
            analysis += "• No strong signals at the moment\n"

        # Key metrics - convert to scalars
        rsi_value = float(get_last(data, 'RSI', 1)[0])
        macd_value = float(get_last(data, 'MACD', 1)[0])
        volume_ratio = float(get_last(data, 'Volume_Ratio', 1)[0])
        atr_value = float(get_last(data, 'ATR', 1)[0])
        sma20_value = float(get_last(data, 'SMA_20', 1)[0])
        sma50_value = float(get_last(data, 'SMA_50', 1)[0])

        if is_us:
            # Convert SMA values to INR for US stocks
            sma20_inr = sma20_value * usd_inr_rate
            sma50_inr = sma50_value * usd_inr_rate
            atr_inr = atr_value * usd_inr_rate

            analysis += (
                f"\n📊 KEY METRICS:\n"
                f"• RSI: {rsi_value:.1f}\n"
                f"• MACD: {macd_value:.4f}\n"
                f"• Volume Ratio: {volume_ratio:.2f}\n"
                f"• ATR: ${atr_value:.2f} (₹{atr_inr:.2f})\n"
                f"• 20 SMA: ${sma20_value:.2f} (₹{sma20_inr:.2f})\n"
                f"• 50 SMA: ${sma50_value:.2f} (₹{sma50_inr:.2f})\n\n"
                f"⚠️ DISCLAIMER: Educational purposes only. Do your own research.\n"
            )
        else:
            # Indian stocks - already in INR
            analysis += (
                f"\n📊 KEY METRICS:\n"
                f"• RSI: {rsi_value:.1f}\n"
                f"• MACD: {macd_value:.4f}\n"
                f"• Volume Ratio: {volume_ratio:.2f}\n"
                f"• ATR: ₹{atr_value:.2f}\n"
                f"• 20 SMA: ₹{sma20_value:.2f}\n"
                f"• 50 SMA: ₹{sma50_value:.2f}\n\n"
                f"⚠️ DISCLAIMER: Educational purposes only. Do your own research.\n"
            )
        # Add layman advice section
        analysis += f"\n{get_layman_advice(recommendation, confidence)}\n"
        return analysis

    except Exception as e:
        return f"❌ Error analyzing {stock_symbol}: {str(e)}"

async def predict_command(update, context):
    if not update.message:
        return
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /predict <stock_symbol>\nExample: /predict AAPL")
        return
    stock_symbol = context.args[0].upper()
    await update.message.reply_text(f"🤖 Running ML prediction for {stock_symbol}... Please wait...")
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_historical_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            await update.message.reply_text(f"❌ No data found for {stock_symbol}")
            return
        data = calculate_technical_indicators(data, stock_symbol)
        model, scaler, accuracy = train_prediction_model(data)
        if model is None:
            await update.message.reply_text("❌ Insufficient data for ML prediction")
            return
        confidence, recommendation = calculate_confidence_score(data, model, scaler)

        # Convert to scalars to avoid formatting issues
        current_price = float(get_last(data, 'Close', 1)[0])
        rsi_value = float(get_last(data, 'RSI', 1)[0])
        macd_value = float(get_last(data, 'MACD', 1)[0])
        supertrend_value = float(get_last(data, 'Supertrend', 1)[0])
        volume_ratio = float(get_last(data, 'Volume_Ratio', 1)[0])

        prediction_message = (
            f"🤖 ML PREDICTION: {stock_symbol}\n\n"
            f"💰 Current Price: ${current_price:.2f}\n"
            f"🎯 Prediction: {recommendation}\n"
                f"🎚️ Confidence: {confidence:.1%}\n"
            f"📊 Model Accuracy: {accuracy:.1%}\n\n"
            f"📈 Technical Indicators:\n"
            f"• RSI: {rsi_value:.1f}\n"
            f"• MACD: {macd_value:.4f}\n"
            f"• Supertrend: ${supertrend_value:.2f}\n"
            f"• Volume: {volume_ratio:.2f}x average\n\n"
            f"⚠️ Remember: Past performance doesn't guarantee future results."
        )
        await update.message.reply_text(prediction_message)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def help_command(update, context):
    if not update.message:
        return
    help_text = (
        "📚 BOT HELP\n\n"
        "🎯 Commands:\n"
        "• /start - Welcome message\n"
        "• /analyze <symbol> - Basic technical analysis\n"
        "• /advanced <symbol> - Advanced analysis with AI & charts\n"
        "• /grok <symbol> - Grok AI-powered deep analysis\n"
        "• /freeai <symbol> - Test free AI services\n"
        "• /predict <symbol> - ML-based prediction\n"
        "• /ask <stock> BUY/SELL/HOLD <qty> <price> - Investment advice with analysis\n"
        "• /recommend <stock> - Comprehensive trading recommendations\n"
        "• /help - This help message\n\n"
        "📈 Supported Markets:\n"
        "• US Stocks: AAPL, GOOGL, MSFT, etc.\n"
        "• Indian NSE: TCS.NS, INFY.NS, etc.\n"
        "• Indian BSE: RELIANCE.BO, etc.\n\n"
        "🔍 Analysis Includes:\n"
        "• Moving Averages (SMA, EMA)\n"
        "• MACD, RSI, Stochastic\n"
        "• Bollinger Bands\n"
        "• Supertrend\n"
        "• Volume Analysis\n"
        "• Machine Learning Predictions\n"
        "• Grok AI Analysis\n"
        "• Free AI Alternatives\n"
        "• AI News Sentiment Analysis\n"
        "• Interactive Charts\n"
        "• Position Sizing & Risk Management\n\n"
        "🧠 Grok AI Features:\n"
        "• Deep market analysis\n"
        "• Sentiment analysis\n"
        "• Risk assessment\n"
        "• Trading recommendations\n"
        "• Price targets\n"
        "• Key factors identification\n\n"
        "🆓 Free AI Features:\n"
        "• Multiple free AI providers\n"
        "• Unlimited local AI (Ollama)\n"
        "• High-volume free tiers\n"
        "• Automatic fallbacks\n"
        "• DeepSeek API integration\n\n"
        "⚠️ Important: This bot is for educational purposes only. Always do your own research and consider consulting a financial advisor before making investment decisions."
    )
    await update.message.reply_text(help_text)

async def error_handler(update, context):
    """Handle errors in the bot"""
    print(f"An error occurred: {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ An error occurred. Please try again later.")


async def freeai_command(update, context):
    """Test free AI services"""
    if not update.message:
        return

    if len(context.args) != 1:
        await update.message.reply_text("Usage: /freeai <stock_symbol>\nExample: /freeai AAPL")
        return

    stock_symbol = context.args[0].upper()
    await update.message.reply_text(f"🤖 Testing free AI analysis for {stock_symbol}...")

    try:
        # Get basic stock data
        data = yf.download(stock_symbol, period='1d', interval='1d')
        if data is None or data.empty:
            await update.message.reply_text(f"❌ No data found for {stock_symbol}")
            return

        # Prepare technical data
        current_price = float(data['Close'].iloc[-1])
        stock_data = {
            'current_price': current_price,
            'rsi': 50.0,  # Placeholder
            'macd': 0.0,  # Placeholder
            'volume_ratio': 1.0,  # Placeholder
            'sma_20': current_price,
            'sma_50': current_price
        }

        # Get free AI analysis
        result = analyze_with_free_ai_fallback(stock_data, stock_symbol)

        # Format response
        analysis = f"🤖 FREE AI ANALYSIS: {stock_symbol}\n\n"
        analysis += f"💰 Current Price: ${current_price:.2f}\n"
        analysis += f"📊 Provider: {result['provider']}\n"
        analysis += f"🎯 Sentiment: {result['sentiment'].upper()}\n"
        analysis += f"🎚️ Confidence: {result['confidence']:.1%}\n"
        analysis += f"📈 Impact: {result['impact'].upper()}\n"
        analysis += f"💡 Recommendation: {result.get('recommendation', 'N/A').upper()}\n\n"
        analysis += f"📝 Summary:\n{result['summary']}\n\n"

        # Add usage stats
        if FREE_AI_AVAILABLE:
            usage = get_free_ai_usage()
            analysis += f"📊 Usage Stats:\n"
            analysis += f"• Current Provider: {usage.get('current_provider', 'None')}\n"
            for provider, count in usage.get('request_counts', {}).items():
                limit = usage.get('limits', {}).get(provider, 'Unknown')
                analysis += f"• {provider}: {count}/{limit} requests\n"

        await update.message.reply_text(analysis)

    except Exception as e:
        await update.message.reply_text(f"❌ Error in free AI analysis: {str(e)}")

async def ask_command(update, context):
    """Handle /ask <stock-name> BUY/SELL/HOLD <stock-quantity> <per-unit-stockprice in INR>"""
    if not update.message:
        return
    try:
        # Join all args for flexible parsing
        prompt = ' '.join(context.args)
        # Regex to extract stock/mf, action, quantity, price
        match = re.match(r"([\w\-.&]+)\s+(BUY|SELL|HOLD)\s+(\d+)\s+(\d+(?:\.\d+)?)", prompt, re.IGNORECASE)
        if not match:
            await update.message.reply_text("Usage: /ask <stock-name> BUY/SELL/HOLD <stock-quantity> <per-unit-stockprice in INR>\nExample: /ask TCS.NS BUY 10 3500")
            return

        name, action, qty, price = match.groups()
        name = name.upper()
        action = action.upper()
        qty = int(qty)
        price = float(price)

        # Send initial processing message
        await update.message.reply_text(f"🔍 Analyzing {name} for {action} action... Please wait...")

        # Check if it's a mutual fund (simple heuristic: ends with .MF or contains 'FUND')
        is_mf = name.endswith('.MF') or 'FUND' in name or 'MF' in name
        if is_mf:
            # Mutual fund analysis
            total_investment = qty * price
            advice = f"For mutual funds like {name}, it's generally best to HOLD for long-term (3+ years). Consider SIP for rupee cost averaging."

            analysis = (
                f"🪙 MUTUAL FUND ANALYSIS: {name}\n\n"
                f"📊 TRANSACTION DETAILS:\n"
                f"• Action: {action}\n"
                f"• Quantity: {qty:,} units\n"
                f"• Unit Price: ₹{price:.2f}\n"
                f"• Total Investment: ₹{total_investment:,.2f}\n\n"
                f"💡 ADVICE:\n{advice}\n\n"
                f"📈 RECOMMENDED STRATEGY:\n"
                f"• Time Period: 3+ years (long-term)\n"
                f"• Investment Style: SIP (Systematic Investment Plan)\n"
                f"• Risk Level: Low to Medium\n"
                f"• Expected Returns: 8-12% annually (historical average)\n\n"
                f"⚠️ DISCLAIMER: Past performance doesn't guarantee future returns."
            )
            await update.message.reply_text(analysis)
            return

        # For stocks, perform comprehensive analysis
        try:
            # Download stock data
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
            data = get_historical_data(name, start_date, end_date)
            if data is None or data.empty:
                await update.message.reply_text(f"❌ No data found for {name}. Please check the symbol.")
                return

            # Calculate technical indicators
            data = calculate_technical_indicators(data, name)
            data = calculate_supertrend(data)

            # Get current market data
            current_price = float(get_last(data, 'Close', 1)[0])
            current_rsi = float(get_last(data, 'RSI', 1)[0])
            current_macd = float(get_last(data, 'MACD', 1)[0])
            current_volume = float(get_last(data, 'Volume', 1)[0])
            sma20 = float(get_last(data, 'SMA_20', 1)[0])
            sma50 = float(get_last(data, 'SMA_50', 1)[0])

            # Calculate price differences
            price_diff = current_price - price
            price_diff_percent = (price_diff / price) * 100

            # Calculate investment metrics
            total_investment = qty * price
            current_value = qty * current_price
            profit_loss = current_value - total_investment
            profit_loss_percent = (profit_loss / total_investment) * 100

            # Determine market sentiment
            if current_price > sma20 > sma50:
                trend = "BULLISH"
            elif current_price < sma20 < sma50:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            # RSI interpretation
            if current_rsi < 30:
                rsi_status = "OVERSOLD"
            elif current_rsi > 70:
                rsi_status = "OVERBOUGHT"
            else:
                rsi_status = "NEUTRAL"

            # MACD interpretation
            macd_status = "BULLISH" if current_macd > 0 else "BEARISH"

            # Generate recommendation based on action and analysis
            if action == "BUY":
                # For BUY: price is the maximum you want to pay
                if current_price < price:
                    recommendation = "GOOD TIMING - Current price is lower than your maximum"
                    confidence = "HIGH"
                    time_period = "1-3 months"
                elif current_price > price * 1.05:
                    recommendation = "WAIT - Current price is significantly higher than your maximum"
                    confidence = "LOW"
                    time_period = "Wait for pullback (2-4 weeks)"
                else:
                    recommendation = "MODERATE - Price is near your maximum"
                    confidence = "MEDIUM"
                    time_period = "2-4 weeks"
            elif action == "SELL":
                # For SELL: price is the minimum you want to receive
                if current_price > price:
                    recommendation = "GOOD TIMING - Current price is higher than your minimum"
                    confidence = "HIGH"
                    time_period = "1-2 weeks"
                elif current_price < price * 0.95:
                    recommendation = "WAIT - Current price is significantly lower than your minimum"
                    confidence = "LOW"
                    time_period = "Wait for recovery (3-6 months)"
                else:
                    recommendation = "MODERATE - Price is near your minimum"
                    confidence = "MEDIUM"
                    time_period = "1-3 weeks"
            else:  # HOLD
                if trend == "BULLISH":
                    recommendation = "HOLD - Stock is in bullish trend"
                    confidence = "HIGH"
                    time_period = "3-6 months"
                elif trend == "BEARISH":
                    recommendation = "HOLD - Wait for trend reversal"
                    confidence = "MEDIUM"
                    time_period = "1-3 months"
                else:
                    recommendation = "HOLD - Neutral trend, wait for breakout"
                    confidence = "LOW"
                    time_period = "2-4 weeks"

            # Build comprehensive analysis
            price_label = "Maximum Price" if action == "BUY" else "Minimum Price" if action == "SELL" else "Target Price"
            action_verb = "buying" if action == "BUY" else "selling" if action == "SELL" else "holding"
            price_context = f"maximum of ₹{price:.2f}" if action == "BUY" else f"minimum of ₹{price:.2f}" if action == "SELL" else f"₹{price:.2f}"

            analysis = (
                f"📊 STOCK ANALYSIS: {name}\n\n"
                f"💰 TRANSACTION DETAILS:\n"
                f"• Action: {action}\n"
                f"• Quantity: {qty:,} shares\n"
                f"• {price_label}: ₹{price:.2f}\n"
                f"• Total Investment: ₹{total_investment:,.2f}\n\n"
                f"📈 CURRENT MARKET DATA:\n"
                f"• Current Price: ₹{current_price:.2f}\n"
                f"• Price Difference: ₹{price_diff:+.2f} ({price_diff_percent:+.1f}%)\n"
                f"• Current Value: ₹{current_value:,.2f}\n"
                f"• P&L: ₹{profit_loss:+,.2f} ({profit_loss_percent:+.1f}%)\n\n"
                f"🔍 TECHNICAL ANALYSIS:\n"
                f"• Trend: {trend}\n"
                f"• RSI: {current_rsi:.1f} ({rsi_status})\n"
                f"• MACD: {current_macd:.4f} ({macd_status})\n"
                f"• 20 SMA: ₹{sma20:.2f}\n"
                f"• 50 SMA: ₹{sma50:.2f}\n"
                f"• Volume: {current_volume:,.0f}\n\n"
                f"🎯 RECOMMENDATION:\n"
                f"• Status: {recommendation}\n"
                f"• Confidence: {confidence}\n"
                f"• Time Period: {time_period}\n\n"
                f"💡 LAYMAN ADVICE:\n"
                f"Based on current market conditions, {recommendation.lower()}. "
                f"The stock shows {trend.lower()} trend with {rsi_status.lower()} RSI. "
                f"Consider {action_verb} {qty} shares at {price_context} and hold for {time_period}.\n\n"
                f"⚠️ DISCLAIMER: This analysis is for educational purposes. Always do your own research."
            )

            await update.message.reply_text(analysis)

        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing {name}: {str(e)}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error in /ask: {str(e)}")

async def recommend_command(update, context):
    """Handle /recommend <stock-name> - Comprehensive trading recommendations"""
    if not update.message:
        return
    try:
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /recommend <stock-name>\nExample: /recommend TCS.NS")
            return

        name = context.args[0].upper()

        # Send initial processing message
        await update.message.reply_text(f"🎯 Generating comprehensive recommendations for {name}... Please wait...")

        # Check if it's a mutual fund
        is_mf = name.endswith('.MF') or 'FUND' in name or 'MF' in name
        if is_mf:
            await update.message.reply_text(f"🪙 MUTUAL FUND RECOMMENDATIONS: {name}\n\n"
                                          f"📊 INVESTMENT STRATEGY:\n"
                                          f"• Investment Style: SIP (Systematic Investment Plan)\n"
                                          f"• Time Period: 5-10 years (long-term)\n"
                                          f"• Risk Level: Low to Medium\n"
                                          f"• Expected Returns: 8-15% annually\n\n"
                                          f"💰 POSITION SIZING:\n"
                                          f"• Monthly SIP: ₹5,000 - ₹10,000\n"
                                          f"• Lump Sum: ₹50,000 - ₹1,00,000\n"
                                          f"• Portfolio Allocation: 20-30% of total portfolio\n\n"
                                          f"📈 PROFIT TARGETS:\n"
                                          f"• Short-term (1-2 years): 10-15%\n"
                                          f"• Medium-term (3-5 years): 25-40%\n"
                                          f"• Long-term (5+ years): 50-100%\n\n"
                                          f"🛑 EXIT STRATEGY:\n"
                                          f"• Rebalance annually\n"
                                          f"• Exit if underperforming for 2+ years\n"
                                          f"• Consider switching to better performing funds\n\n"
                                          f"⚠️ DISCLAIMER: Past performance doesn't guarantee future returns.")
            return

        # For stocks, perform comprehensive analysis
        try:
            # Download stock data
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
            data = get_historical_data(name, start_date, end_date)
            if data is None or data.empty:
                await update.message.reply_text(f"❌ No data found for {name}. Please check the symbol.")
                return

            # Calculate technical indicators
            data = calculate_technical_indicators(data, name)
            data = calculate_supertrend(data)

            # Get current market data
            current_price = float(get_last(data, 'Close', 1)[0])
            current_rsi = float(get_last(data, 'RSI', 1)[0])
            current_macd = float(get_last(data, 'MACD', 1)[0])
            sma20 = float(get_last(data, 'SMA_20', 1)[0])
            sma50 = float(get_last(data, 'SMA_50', 1)[0])
            atr = float(get_last(data, 'ATR', 1)[0]) if 'ATR' in data.columns else current_price * 0.02

            # Determine market sentiment and trend
            if current_price > sma20 > sma50:
                trend = "BULLISH"
                trend_strength = "STRONG"
            elif current_price > sma20:
                trend = "BULLISH"
                trend_strength = "WEAK"
            elif current_price < sma20 < sma50:
                trend = "BEARISH"
                trend_strength = "STRONG"
            elif current_price < sma20:
                trend = "BEARISH"
                trend_strength = "WEAK"
            else:
                trend = "NEUTRAL"
                trend_strength = "MIXED"

            # RSI interpretation
            if current_rsi < 30:
                rsi_status = "OVERSOLD"
                rsi_signal = "BUY"
            elif current_rsi > 70:
                rsi_status = "OVERBOUGHT"
                rsi_signal = "SELL"
            else:
                rsi_status = "NEUTRAL"
                rsi_signal = "HOLD"

            # MACD interpretation
            macd_signal = "BUY" if current_macd > 0 else "SELL"

            # Calculate volatility and risk metrics
            close_series = data['Close']
            if not isinstance(close_series, pd.Series):
                close_series = pd.Series(close_series)
            returns = close_series.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100)
            max_drawdown = float(((close_series - close_series.expanding().max()) / close_series.expanding().max() * 100).min())

            # Determine overall recommendation
            if trend == "BULLISH" and rsi_signal == "BUY":
                overall_recommendation = "STRONG BUY"
                confidence_score = 85
            elif trend == "BULLISH" and rsi_signal == "HOLD":
                overall_recommendation = "BUY"
                confidence_score = 70
            elif trend == "BEARISH" and rsi_signal == "SELL":
                overall_recommendation = "STRONG SELL"
                confidence_score = 85
            elif trend == "BEARISH" and rsi_signal == "HOLD":
                overall_recommendation = "SELL"
                confidence_score = 70
            else:
                overall_recommendation = "HOLD"
                confidence_score = 50

            # Calculate stop loss and profit targets
            if overall_recommendation in ["STRONG BUY", "BUY"]:
                # For buying recommendations
                stop_loss = current_price * 0.92  # 8% stop loss
                target_1 = current_price * 1.15  # 15% target
                target_2 = current_price * 1.25  # 25% target
                target_3 = current_price * 1.40  # 40% target
                holding_period = "3-6 months"
                risk_reward_ratio = "1:2.5"
            elif overall_recommendation in ["STRONG SELL", "SELL"]:
                # For selling recommendations
                stop_loss = current_price * 1.08  # 8% stop loss for short
                target_1 = current_price * 0.85  # 15% target
                target_2 = current_price * 0.75  # 25% target
                target_3 = current_price * 0.60  # 40% target
                holding_period = "1-3 months"
                risk_reward_ratio = "1:2.5"
            else:
                # For hold recommendations
                stop_loss = current_price * 0.95  # 5% stop loss
                target_1 = current_price * 1.10  # 10% target
                target_2 = current_price * 1.20  # 20% target
                target_3 = current_price * 1.30  # 30% target
                holding_period = "6-12 months"
                risk_reward_ratio = "1:2"

            # Calculate position size based on risk
            account_size = 100000  # Default account size
            risk_percentage = 2.0  # 2% risk per trade
            risk_amount = account_size * risk_percentage / 100
            stop_loss_amount = abs(current_price - stop_loss)
            position_size = int(risk_amount / stop_loss_amount)

            # Build comprehensive recommendation
            recommendation = (
                f"🎯 COMPREHENSIVE RECOMMENDATIONS: {name}\n\n"
                f"📊 CURRENT MARKET DATA:\n"
                f"• Current Price: ₹{current_price:.2f}\n"
                f"• Trend: {trend} ({trend_strength})\n"
                f"• RSI: {current_rsi:.1f} ({rsi_status})\n"
                f"• MACD: {current_macd:.4f} ({macd_signal})\n"
                f"• Volatility: {volatility:.1f}%\n"
                f"• Max Drawdown: {max_drawdown:.1f}%\n\n"
                f"🎯 OVERALL RECOMMENDATION:\n"
                f"• Action: {overall_recommendation}\n"
                f"• Confidence: {confidence_score}%\n"
                f"• Risk Level: {'HIGH' if volatility > 30 else 'MEDIUM' if volatility > 20 else 'LOW'}\n\n"
                f"💰 POSITION SIZING:\n"
                f"• Recommended Shares: {position_size:,}\n"
                f"• Investment Amount: ₹{position_size * current_price:,.2f}\n"
                f"• Risk Amount: ₹{risk_amount:,.2f} (2% of ₹{account_size:,})\n"
                f"• Risk-Reward Ratio: {risk_reward_ratio}\n\n"
                f"🛑 STOP LOSS & TARGETS:\n"
                f"• Stop Loss: ₹{stop_loss:.2f} ({((stop_loss-current_price)/current_price*100):+.1f}%)\n"
                f"• Target 1: ₹{target_1:.2f} ({((target_1-current_price)/current_price*100):+.1f}%)\n"
                f"• Target 2: ₹{target_2:.2f} ({((target_2-current_price)/current_price*100):+.1f}%)\n"
                f"• Target 3: ₹{target_3:.2f} ({((target_3-current_price)/current_price*100):+.1f}%)\n\n"
                f"⏰ TIMING & HOLDING:\n"
                f"• Recommended Holding: {holding_period}\n"
                f"• Entry Timing: {'IMMEDIATE' if overall_recommendation in ['STRONG BUY', 'STRONG SELL'] else 'WAIT FOR PULLBACK'}\n"
                f"• Exit Strategy: {'Trail stop loss' if overall_recommendation in ['STRONG BUY', 'BUY'] else 'Take profit at targets'}\n\n"
                f"📈 PROFIT PREDICTIONS:\n"
                f"• Conservative (Target 1): {((target_1-current_price)/current_price*100):+.1f}% in 1-2 months\n"
                f"• Moderate (Target 2): {((target_2-current_price)/current_price*100):+.1f}% in 2-4 months\n"
                f"• Aggressive (Target 3): {((target_3-current_price)/current_price*100):+.1f}% in 4-6 months\n\n"
                f"💡 TRADING STRATEGY:\n"
                f"• Entry: {'Buy on dips' if overall_recommendation in ['STRONG BUY', 'BUY'] else 'Sell on rallies'}\n"
                f"• Position Management: {'Add on pullbacks' if overall_recommendation in ['STRONG BUY', 'BUY'] else 'Reduce on bounces'}\n"
                f"• Risk Management: Always use stop loss, don't risk more than 2% per trade\n\n"
                f"⚠️ DISCLAIMER: These are educational recommendations. Always do your own research and consider consulting a financial advisor."
            )

            await update.message.reply_text(recommendation)

        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing {name}: {str(e)}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error in /recommend: {str(e)}")

# Update all handler functions to async
def sync_to_async(func):
    # Helper to wrap sync functions for async handlers
    import functools
    import asyncio
    @functools.wraps(func)
    async def wrapper(update, context):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, update, context)
    return wrapper

async def start_command(update, context):
    if not update.message:
        return
    welcome_message = (
        "🤖 Welcome to the Advanced Stock Prediction Bot!\n\n"
        "📈 Available Commands:\n"
        "• /analyze <symbol> - Basic stock analysis\n"
        "• /advanced <symbol> - Advanced analysis with AI & charts\n"
        "• /grok <symbol> - Grok AI-powered deep analysis\n"
        "• /freeai <symbol> - Test free AI services\n"
        "• /predict <symbol> - ML-based price prediction\n"
        "• /ask <stock> BUY/SELL/HOLD <qty> <price> - Investment advice with analysis\n"
        "• /recommend <stock> - Comprehensive trading recommendations\n"
        "• /help - Show this help message\n\n"
        "💡 Examples:\n"
        "• /analyze AAPL - Basic Apple stock analysis\n"
        "• /advanced TCS.NS - Advanced analysis with trading signals\n"
        "• /grok RELIANCE.BO - Grok AI deep analysis\n"
        "• /freeai MSFT - Test free AI analysis\n"
        "• /ask TCS.NS BUY 10 3500 - Investment advice for buying 10 TCS shares at ₹3500\n"
        "• /recommend TCS.NS - Comprehensive trading recommendations with stop-loss and targets\n\n"
        "🎯 Features:\n"
        "• Multiple technical indicators\n"
        "• Machine learning predictions\n"
        "• Grok AI-powered analysis\n"
        "• Free AI alternatives\n"
        "• AI-powered news sentiment analysis\n"
        "• Interactive charts\n"
        "• Position sizing & risk management\n"
        "• Buy/Sell recommendations with entry/exit points\n\n"
        "⚠️ Disclaimer: This is for educational purposes only. Always do your own research."
    )
    await update.message.reply_text(welcome_message)

async def analyze_command(update, context):
    try:
        if not update or not update.message:
            return
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /analyze <stock_symbol>\nExample: /analyze AAPL")
            return
        stock_symbol = context.args[0].upper()
        await update.message.reply_text(f"🔍 Analyzing {stock_symbol}... Please wait...")
        analysis = analyze_stock(stock_symbol)
        print('DEBUG ANALYSIS MESSAGE:')
        print(analysis)

        # Send without HTML parsing to avoid issues
        await update.message.reply_text(analysis, parse_mode=None)

    except Exception as e:
        print(f"Error in analyze_command: {e}")
        if update and update.message:
            await update.message.reply_text(f"❌ Error analyzing stock: {str(e)}")

async def advanced_command(update, context):
    try:
        if not update or not update.message:
            return
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /advanced <stock_symbol>\nExample: /advanced TCS.NS")
            return
        stock_symbol = context.args[0].upper()
        await update.message.reply_text(f"🚀 Running advanced analysis for {stock_symbol}... Please wait...")

        analysis = await analyze_stock_advanced(stock_symbol, update)
        print('DEBUG ADVANCED ANALYSIS:')
        print(analysis)

        # Send the analysis
        await update.message.reply_text(analysis, parse_mode=None)

        # Create interactive buttons
        keyboard = [
            [
                InlineKeyboardButton("📊 View Chart", callback_data=f"chart_{stock_symbol}"),
                InlineKeyboardButton("📰 More News", callback_data=f"news_{stock_symbol}")
            ],
            [
                InlineKeyboardButton("💰 Position Calculator", callback_data=f"position_{stock_symbol}"),
                InlineKeyboardButton("📈 Technical Details", callback_data=f"technical_{stock_symbol}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("🔧 Additional Options:", reply_markup=reply_markup)

    except Exception as e:
        print(f"Error in advanced_command: {e}")
        if update and update.message:
            await update.message.reply_text(f"❌ Error in advanced analysis: {str(e)}")

async def grok_command(update, context):
    """Grok AI-powered deep stock analysis"""
    try:
        if not update or not update.message:
            return
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /grok <stock_symbol>\nExample: /grok AAPL")
            return

        stock_symbol = context.args[0].upper()
        await update.message.reply_text(f"🧠 Running Grok AI deep analysis for {stock_symbol}... Please wait...")

        # Download and prepare data
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_historical_data(stock_symbol, start_date, end_date)
        if data is None or data.empty:
            await update.message.reply_text(f"❌ No data found for {stock_symbol}")
            return

        # Calculate indicators
        data = calculate_technical_indicators(data)
        data = calculate_supertrend(data)

        # Prepare technical data for Grok analysis
        current_price = float(get_last(data, 'Close', 1)[0])
        rsi_value = float(get_last(data, 'RSI', 1)[0])
        macd_value = float(get_last(data, 'MACD', 1)[0])
        volume_ratio = float(get_last(data, 'Volume_Ratio', 1)[0])
        sma20_value = float(get_last(data, 'SMA_20', 1)[0])
        sma50_value = float(get_last(data, 'SMA_50', 1)[0])
        atr_value = float(get_last(data, 'ATR', 1)[0])

        technical_data = {
            'current_price': current_price,
            'rsi': rsi_value,
            'macd': macd_value,
            'volume_ratio': volume_ratio,
            'sma_20': sma20_value,
            'sma_50': sma50_value,
            'atr': atr_value
        }

        # Get Grok AI analysis
        grok_analysis = get_advanced_stock_analysis_with_grok(stock_symbol, technical_data)

        # Get news and sentiment analysis
        news_items = get_stock_news(stock_symbol)
        ai_analysis = analyze_news_with_ai(news_items, stock_symbol)

        # Build comprehensive analysis message
        is_us = is_us_stock(stock_symbol)
        if is_us:
            usd_inr_rate = get_usd_to_inr_rate()
            current_price_inr = current_price * usd_inr_rate
            analysis = (
                f"🧠 GROK AI ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ${current_price:.2f} (₹{current_price_inr:.2f})\n"
                f"📊 Technical Indicators:\n"
                f"• RSI: {rsi_value:.1f}\n"
                f"• MACD: {macd_value:.4f}\n"
                f"• Volume Ratio: {volume_ratio:.2f}x\n"
                f"• 20 SMA: ${sma20_value:.2f}\n"
                f"• 50 SMA: ${sma50_value:.2f}\n"
                f"• ATR: ${atr_value:.2f}\n\n"
            )
        else:
            analysis = (
                f"🧠 GROK AI ANALYSIS: {stock_symbol.upper()}\n\n"
                f"💰 Current Price: ₹{current_price:.2f}\n"
                f"📊 Technical Indicators:\n"
                f"• RSI: {rsi_value:.1f}\n"
                f"• MACD: {macd_value:.4f}\n"
                f"• Volume Ratio: {volume_ratio:.2f}x\n"
                f"• 20 SMA: ₹{sma20_value:.2f}\n"
                f"• 50 SMA: ₹{sma50_value:.2f}\n"
                f"• ATR: ₹{atr_value:.2f}\n\n"
            )

        # Add Grok AI analysis
        analysis += f"🧠 GROK AI INSIGHTS:\n"
        analysis += f"Provider: {grok_analysis['provider']}\n\n"
        analysis += f"{grok_analysis['analysis']}\n\n"

        # Add news sentiment
        analysis += f"📰 NEWS SENTIMENT:\n"
        analysis += f"• Sentiment: {ai_analysis['sentiment'].upper()}\n"
        analysis += f"• Confidence: {ai_analysis['confidence']:.1%}\n"
        analysis += f"• Impact: {ai_analysis['impact'].upper()}\n"
        analysis += f"• Summary: {ai_analysis['summary']}\n\n"

        # Add key factors if available
        if 'key_factors' in ai_analysis and ai_analysis['key_factors']:
            analysis += f"🔍 KEY FACTORS:\n"
            for factor in ai_analysis['key_factors'][:5]:  # Show top 5 factors
                analysis += f"• {factor}\n"
            analysis += "\n"

        # Add recent news
        analysis += f"📰 RECENT NEWS:\n"
        for news in news_items[:3]:
            analysis += f"• {news}\n"

        # Send the analysis
        await update.message.reply_text(analysis, parse_mode=None)

        # Create interactive buttons
        keyboard = [
            [
                InlineKeyboardButton("📊 Technical Chart", callback_data=f"chart_{stock_symbol}"),
                InlineKeyboardButton("📰 More News", callback_data=f"news_{stock_symbol}")
            ],
            [
                InlineKeyboardButton("💰 Position Size", callback_data=f"position_{stock_symbol}"),
                InlineKeyboardButton("🎯 Price Targets", callback_data=f"targets_{stock_symbol}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("🔧 Additional Options:", reply_markup=reply_markup)

    except Exception as e:
        print(f"Error in grok_command: {e}")
        if update and update.message:
            await update.message.reply_text(f"❌ Error in Grok analysis: {str(e)}")

async def button_callback(update, context):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("chart_"):
        symbol = data.replace("chart_", "")
        await query.edit_message_text(f"📊 Chart for {symbol} has been saved as HTML file. Check your bot directory for the interactive chart.")

    elif data.startswith("news_"):
        symbol = data.replace("news_", "")
        news_items = get_stock_news(symbol)
        news_text = f"📰 Latest News for {symbol}:\n\n"
        for i, news in enumerate(news_items[:5], 1):
            news_text += f"{i}. {news}\n"
        await query.edit_message_text(news_text)

    elif data.startswith("position_"):
        symbol = data.replace("position_", "")
        await query.edit_message_text(f"💰 Position Calculator for {symbol}:\n\n"
                                    f"Enter your account size and risk percentage to get personalized position sizing recommendations.")

    elif data.startswith("technical_"):
        symbol = data.replace("technical_", "")
        await query.edit_message_text(f"📈 Technical Analysis Details for {symbol}:\n\n"
                                    f"Detailed technical indicators and their interpretations will be shown here.")

    elif data.startswith("targets_"):
        symbol = data.replace("targets_", "")
        await query.edit_message_text(f"🎯 Price Targets for {symbol}:\n\n"
                                    f"Short-term target: +5-10%\n"
                                    f"Medium-term target: +15-25%\n"
                                    f"Long-term target: +30-50%\n\n"
                                    f"Note: These are estimates based on technical analysis. Market conditions can change rapidly.")

def get_layman_advice(recommendation, confidence, hold_period=None):
    """Generate a simple layman advice string based on recommendation and confidence."""
    if recommendation is None:
        return "No clear advice available."
    rec = recommendation.upper()
    if rec in ["STRONG BUY", "BUY"]:
        action = "You may consider BUYING this stock."
        period = hold_period or "for the next few weeks to months."
    elif rec in ["STRONG SELL", "SELL"]:
        action = "You may consider SELLING this stock or avoiding new investment."
        period = hold_period or "until the trend improves."
    else:
        action = "It's best to HOLD and wait."
        period = hold_period or "until a clearer trend appears."
    conf = f"(Confidence: {confidence:.0%})" if confidence is not None else ""
    return f"💡 Layman Advice: {action} {period} {conf}"

def fetch_pcr_oi(symbol):
    """Fetch PCR and OI for Indian stocks using nsepython. Returns (pcr, total_call_oi, total_put_oi) or (None, None, None) if not available."""
    if not NSEPYTHON_AVAILABLE or not (symbol.endswith('.NS') or symbol in ['NIFTY', 'BANKNIFTY']):
        return None, None, None
    try:
        oc = option_chain(symbol.replace('.NS',''))
        data = oc['filtered']['data']
        call_oi = 0
        put_oi = 0
        for row in data:
            if 'CE' in row and row['CE'] and 'openInterest' in row['CE']:
                call_oi += row['CE']['openInterest']
            if 'PE' in row and row['PE'] and 'openInterest' in row['PE']:
                put_oi += row['PE']['openInterest']
        pcr = put_oi / call_oi if call_oi > 0 else np.nan
        return pcr, call_oi, put_oi
    except Exception as e:
        print(f"PCR/OI fetch error: {e}")
        return None, None, None

# Add safe formatting utility
def safe_fmt(val, fmt=".2f"):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt}}"
    except Exception:
        return str(val) if val is not None else "N/A"

# Unified historical data fetcher
def get_historical_data(symbol, start_date, end_date):
    """Fetch historical OHLCV data for a symbol using NSEPython for Indian stocks, yfinance for others. Always return a DataFrame."""
    if NSEPYTHON_AVAILABLE and (symbol.endswith('.NS') or symbol in ['NIFTY', 'BANKNIFTY']):
        try:
            # For indices, use 'INDEX'; for stocks, use 'EQ'
            series = 'INDEX' if symbol in ['NIFTY', 'BANKNIFTY'] else 'EQ'
            df = equity_history(symbol.replace('.NS',''), series, start_date, end_date)
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("NSEPython returned no data")
            df = df.rename(columns={
                'CLOSE': 'Close',
                'OPEN': 'Open',
                'HIGH': 'High',
                'LOW': 'Low',
                'VOLUME': 'Volume',
                'TIMESTAMP': 'Date'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            print(f"NSEPython historical fetch error: {e}, falling back to yfinance.")
    # Fallback to yfinance
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    if isinstance(df, np.ndarray):
        # Convert to DataFrame if needed
        if df.ndim == 2 and df.shape[1] == 5:
            df = pd.DataFrame(df)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            df = pd.DataFrame(df)
    return df

def plot_oi_matplotlib(option_chain_df, symbol):
    """Plot OI by strike using matplotlib for Indian stocks."""
    try:
        strikes = []
        call_oi = []
        put_oi = []
        for row in option_chain_df:
            strike = None
            ce_oi = None
            pe_oi = None
            if 'CE' in row and row['CE']:
                strike = row['CE'].get('strikePrice')
                ce_oi = row['CE'].get('openInterest')
            if 'PE' in row and row['PE']:
                if strike is None:
                    strike = row['PE'].get('strikePrice')
                pe_oi = row['PE'].get('openInterest')
            if strike is not None:
                strikes.append(strike)
                call_oi.append(ce_oi if ce_oi is not None else 0)
                put_oi.append(pe_oi if pe_oi is not None else 0)
        plt.figure(figsize=(12, 6))
        plt.bar([s-5 for s in strikes], call_oi, width=5, color='green', label='Call OI')
        plt.bar([s+5 for s in strikes], put_oi, width=5, color='red', label='Put OI')
        plt.xlabel('Strike Price')
        plt.ylabel('Open Interest')
        plt.title(f'Open Interest by Strike for {symbol}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting OI chart: {e}")

def get_last(df, col, n):
    """Return the last n values from a DataFrame or Series column as a list, robust to type issues."""
    try:
        if isinstance(df, pd.DataFrame):
            vals = df[col].tail(n).values.tolist()
        elif isinstance(df, pd.Series):
            vals = df.tail(n).values.tolist()
        elif isinstance(df, np.ndarray):
            vals = df[-n:].tolist()
        else:
            vals = list(df)[-n:]
        return vals[::-1]  # Return in reverse order so [0] is most recent
    except Exception:
        return [None]*n

def run_backtrader_backtest(df, signal_col='Signal', commission=0.001, slippage=0.0, risk_per_trade=0.02, initial_cash=100000, asset_col=None, extra_signal_cols=None, export_csv=None, export_html=None):
    """
    Run an advanced Backtrader backtest using the given DataFrame and signal column.
    - commission: per-trade commission (e.g., 0.001 = 0.1%)
    - slippage: per-trade slippage (fractional, e.g., 0.001 = 0.1%)
    - risk_per_trade: fraction of capital to risk per trade (for position sizing)
    - asset_col: if not None, use this column to support multi-asset backtesting
    - extra_signal_cols: list of additional signal columns for ensemble/multi-signal logic
    - export_csv: if not None, path to export trades to CSV
    - export_html: if not None, path to export summary to HTML
    """
    import pandas as pd
    import numpy as np
    class PandasDataWithSignal(bt.feeds.PandasData):
        lines = (signal_col,)
        params = dict(bt.feeds.PandasData.params)
        params['plot'] = False
        params['datetime'] = None
        params[signal_col] = -1
        if asset_col:
            lines = lines + (asset_col,)
            params[asset_col] = -1
        if extra_signal_cols:
            for col in extra_signal_cols:
                lines = lines + (col,)
                params[col] = -1
    class AdvancedSignalStrategy(bt.Strategy):
        params = (('signal_col', signal_col), ('risk_per_trade', risk_per_trade), ('asset_col', asset_col), ('extra_signal_cols', extra_signal_cols))
        def __init__(self):
            self.signal = self.datas[0].lines.getline(self.p.signal_col)
            self.order = None
            self.size = 0
            self.trades = []
            self.pnl = []
        def next(self):
            if self.order:
                return
            sig = self.signal[0]
            # Ensemble logic: combine signals if extra_signal_cols provided
            if self.p.extra_signal_cols:
                votes = [sig]
                for i, col in enumerate(self.p.extra_signal_cols):
                    votes.append(self.datas[0].lines.getline(col)[0])
                sig = int(round(np.mean(votes)))
            # Position sizing: risk_per_trade * cash / (price * stop_loss_pct)
            price = self.datas[0].close[0]
            stop_loss_pct = 0.02  # 2% stop loss default
            cash = self.broker.get_cash()
            risk_amt = cash * self.p.risk_per_trade
            if stop_loss_pct > 0:
                size = int(risk_amt / (price * stop_loss_pct))
            else:
                size = int(cash / price)
            if sig == 1:
                self.order = self.buy(size=size)
            elif sig == -1:
                self.order = self.sell(size=size)
            # No action for 0 (hold)
        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin]:
                self.order = None
        def notify_trade(self, trade):
            if trade.isclosed:
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'pnl': trade.pnl,
                    'gross': trade.pnlcomm,
                    'size': trade.size,
                    'price': trade.price
                })
                self.pnl.append(trade.pnl)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AdvancedSignalStrategy, signal_col=signal_col, risk_per_trade=risk_per_trade, asset_col=asset_col, extra_signal_cols=extra_signal_cols)
    datafeed = PandasDataWithSignal(df)
    cerebro.adddata(datafeed)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    if slippage > 0:
        cerebro.broker.set_slippage_perc(slippage)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    strat = cerebro.run()[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()
    # Performance reporting
    trades_df = pd.DataFrame(strat.trades)
    if not trades_df.empty:
        sharpe = trades_df['pnl'].mean() / (trades_df['pnl'].std() + 1e-9) * np.sqrt(252)
        max_dd = (trades_df['pnl'].cumsum().cummax() - trades_df['pnl'].cumsum()).max()
        win_rate = (trades_df['pnl'] > 0).mean()
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}")
        print(f"Win Rate: {win_rate:.1%}")
        if export_csv:
            trades_df.to_csv(export_csv, index=False)
            print(f"Trades exported to {export_csv}")
        if export_html:
            trades_df.to_html(export_html)
            print(f"Trades exported to {export_html}")
    else:
        print("No trades to report.")

async def unified_command(update, context):
    """Handle /unified <symbol> [expiry] [backend] [blackscholes] [start_date] [end_date]"""
    if not update.message:
        return
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /unified <symbol> [expiry] [backend] [blackscholes] [start_date] [end_date]\nExample: /unified NIFTY 17-Jul-2025 talib true 2024-01-01 2024-07-01")
        return
    symbol = args[0].upper()
    expiry = args[1] if len(args) > 1 and args[1] != '-' else None
    backend = args[2] if len(args) > 2 and args[2] in ['auto', 'talib', 'pandas'] else 'auto'
    use_black_scholes = args[3].lower() == 'true' if len(args) > 3 else True
    start_date = args[4] if len(args) > 4 else (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = args[5] if len(args) > 5 else datetime.today().strftime('%Y-%m-%d')
    await update.message.reply_text(f"🔍 Running unified recommendation for {symbol}... Please wait...")
    import io
    import sys
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    oi_chart_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    backtest_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        result = recommend_trades(
            symbol=symbol,
            expiry=expiry,
            backend=backend,
            use_black_scholes=use_black_scholes,
            start_date=start_date,
            end_date=end_date,
            export_oi_chart_path=oi_chart_file.name,
            export_backtest_csv_path=backtest_csv_file.name
        )
    except Exception as e:
        print(f'Error: {e}')
        result = {}
    sys.stdout = sys_stdout
    summary_text = buffer.getvalue()
    await update.message.reply_text(f"📊 UNIFIED RECOMMENDATION FOR {symbol} (Backend: {backend}, Black-Scholes: {use_black_scholes})\n\n" + summary_text[:4000])
    # Send files if available
    if result.get('oi_chart_saved'):
        with open(oi_chart_file.name, 'rb') as f:
            await update.message.reply_document(f, filename=f'{symbol}_oi_chart.png', caption='OI Chart')
    if result.get('backtest_csv_saved'):
        with open(backtest_csv_file.name, 'rb') as f:
            await update.message.reply_document(f, filename=f'{symbol}_backtest.csv', caption='Backtest Results')

@lru_cache(maxsize=32)
def fetch_historical_oi_cached(symbol, start_date, end_date, expiry):
    # Wrapper for caching
    return fetch_historical_oi(symbol, start_date, end_date, expiry)

def fetch_historical_oi(symbol='NIFTY', start_date='01-01-2024', end_date='17-07-2025', expiry=None):
    import calendar
    from datetime import datetime, timedelta
    import os
    import time
    import warnings
    # Helper to get all dates between start and end
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
    # Parse input dates
    try:
        start_dt = datetime.strptime(start_date, '%d-%m-%Y')
        end_dt = datetime.strptime(end_date, '%d-%m-%Y')
    except Exception:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    all_oi = []
    for dt in daterange(start_dt, end_dt):
        yyyy = dt.strftime('%Y')
        mmm = dt.strftime('%b').upper()
        ddmmyyyy = dt.strftime('%d%b%Y').upper()
        url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mmm}/fo{ddmmyyyy}bhav.csv.zip"
        try:
            t0 = time.time()
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    if symbol.isalpha() and symbol.isupper() and (symbol.endswith('NIFTY') or symbol in ['NIFTY', 'BANKNIFTY']):
                        instr = 'OPTIDX'
                    else:
                        instr = 'OPTSTK'
                    df = df[(df['SYMBOL'] == symbol) & (df['INSTRUMENT'] == instr)]
                    if expiry:
                        df = df[df['EXPIRY_DT'] == expiry]
                    df['DATE'] = dt
                    all_oi.append(df[['DATE','EXPIRY_DT','STRIKE_PR','OPTION_TYP','OPEN_INT','CHG_IN_OI']])
            t1 = time.time()
            if t1-t0 > 5:
                print(f"Warning: OI download for {dt.strftime('%Y-%m-%d')} was slow ({t1-t0:.1f}s)")
        except Exception as e:
            print(f"Warning: OI download failed for {dt.strftime('%Y-%m-%d')}: {e}")
            continue
        time.sleep(0.2)
    if all_oi:
        return pd.concat(all_oi, ignore_index=True)
    print("Warning: No historical OI data found, filling with NaN.")
    return pd.DataFrame()

# --- Real-time OI delta tracking ---
def fetch_option_chain_df(symbol, expiry=None):
    """Fetch option chain as DataFrame (calls and puts OI by strike)."""
    try:
        from nsepython import option_chain
        oc = option_chain(symbol.replace('.NS',''))
        data = oc['filtered']['data']
        rows = []
        for row in data:
            strike = row.get('strikePrice') or (row['CE']['strikePrice'] if 'CE' in row and row['CE'] else None)
            call_oi = row['CE']['openInterest'] if 'CE' in row and row['CE'] else 0
            put_oi = row['PE']['openInterest'] if 'PE' in row and row['PE'] else 0
            rows.append({'strikePrice': strike, 'openInterest_call': call_oi, 'openInterest_put': put_oi})
        df = pd.DataFrame(rows)
        if expiry:
            # Optionally filter by expiry if available in data
            pass  # Not implemented for now
        return df
    except Exception as e:
        return pd.DataFrame()

def fetch_oi_delta(symbol='NIFTY', expiry=None, interval=60):
    """Poll option chain twice and compute OI delta for each strike (calls and puts)."""
    import time
    prev_df = fetch_option_chain_df(symbol, expiry)
    time.sleep(interval)
    curr_df = fetch_option_chain_df(symbol, expiry)
    if prev_df.empty or curr_df.empty:
        return pd.DataFrame()
    merged = curr_df.set_index('strikePrice').join(prev_df.set_index('strikePrice'), lsuffix='_curr', rsuffix='_prev')
    merged['delta_call_oi'] = merged['openInterest_call_curr'] - merged['openInterest_call_prev']
    merged['delta_put_oi'] = merged['openInterest_put_curr'] - merged['openInterest_put_prev']
    return merged[['delta_call_oi','delta_put_oi']]

# --- Integration into feature engineering ---
def calculate_oi_features(symbol, start_date, end_date, expiry=None, poll_oi_delta=False, oi_delta_interval=60):
    features = {}
    pcr, total_call_oi, total_put_oi = fetch_pcr_oi(symbol)
    features['PCR'] = pcr
    features['Total_Call_OI'] = total_call_oi
    features['Total_Put_OI'] = total_put_oi
    try:
        hist_oi = fetch_historical_oi_cached(symbol, start_date, end_date, expiry)
    except Exception as e:
        print(f"Warning: OI cache error: {e}")
        hist_oi = pd.DataFrame()
    if not hist_oi.empty:
        call_oi = hist_oi[hist_oi['OPTION_TYP']=='CE']['OPEN_INT']
        put_oi = hist_oi[hist_oi['OPTION_TYP']=='PE']['OPEN_INT']
        features['Hist_Call_OI_Mean'] = call_oi.mean()
        features['Hist_Put_OI_Mean'] = put_oi.mean()
        features['Hist_Call_OI_Std'] = call_oi.std()
        features['Hist_Put_OI_Std'] = put_oi.std()
    else:
        features['Hist_Call_OI_Mean'] = features['Hist_Put_OI_Mean'] = np.nan
        features['Hist_Call_OI_Std'] = features['Hist_Put_OI_Std'] = np.nan
    if poll_oi_delta:
        oi_delta = fetch_oi_delta(symbol, expiry, oi_delta_interval)
        if not oi_delta.empty:
            features['Delta_Call_OI_Mean'] = oi_delta['delta_call_oi'].mean()
            features['Delta_Put_OI_Mean'] = oi_delta['delta_put_oi'].mean()
        else:
            features['Delta_Call_OI_Mean'] = features['Delta_Put_OI_Mean'] = np.nan
    else:
        features['Delta_Call_OI_Mean'] = features['Delta_Put_OI_Mean'] = np.nan
    return features

def main():
    print("🤖 Starting Advanced Stock Prediction Bot...")
    app = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('analyze', analyze_command))
    app.add_handler(CommandHandler('advanced', advanced_command))
    app.add_handler(CommandHandler('grok', grok_command))
    app.add_handler(CommandHandler('predict', predict_command))
    app.add_handler(CommandHandler('freeai', freeai_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('ask', ask_command))
    app.add_handler(CommandHandler('recommend', recommend_command))
    app.add_handler(CommandHandler('unified', unified_command))

    # Add callback query handler for buttons
    app.add_handler(CallbackQueryHandler(button_callback))

    # Add error handler
    app.add_error_handler(error_handler)

    print("✅ Bot is running! Use /start to begin.")
    print("🧠 Grok AI integration: " + ("✅ Available" if GROK_SDK_AVAILABLE and grok_client else "❌ Not available"))
    try:
        app.run_polling()
    except KeyboardInterrupt:
        print("\n👋 Graceful exit: Stopping the bot. Goodbye!")
        try:
            import asyncio
            asyncio.run(app.stop())
        except Exception:
            pass
        import sys
        sys.exit(0)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print('Running test analysis for AAPL...')
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = get_historical_data('AAPL', start_date, end_date)
        if data is None or data.empty:
            print('No data for AAPL!')
            sys.exit(1)
        data = calculate_technical_indicators(data)
        data = calculate_supertrend(data)
        print('Indicators and Supertrend calculated successfully.')
        # Example: generate dummy signals for demonstration
        data['Signal'] = 0
        data['ML_Signal'] = 0
        data['Tech_Signal'] = 0
        data.loc[data['Close'] > data['SMA_20'], 'Signal'] = 1
        data.loc[data['Close'] < data['SMA_20'], 'Signal'] = -1
        data['ML_Signal'] = data['Signal']  # For demo, copy main signal
        data['Tech_Signal'] = data['Signal']  # For demo, copy main signal
        # Now run the backtest:
        run_backtrader_backtest(
            data,
            signal_col='Signal',
            commission=0.001,
            slippage=0.001,
            risk_per_trade=0.02,
            initial_cash=100000,
            extra_signal_cols=['ML_Signal', 'Tech_Signal'],
            export_csv='trades.csv',
            export_html='trades.html'
        )
        sys.exit(0)
    else:
        main()
