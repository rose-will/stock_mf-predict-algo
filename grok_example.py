#!/usr/bin/env python3
"""
Example script demonstrating Grok API integration for stock analysis.
This shows how to use the Grok AI features independently.
"""

import os
import sys
from typing import List, Dict, Any

# Add current directory to path
sys.path.append('.')

def example_grok_stock_analysis():
    """Example of using Grok AI for stock analysis"""

    # Check if Grok SDK is available
    try:
        from xai_sdk import Client
        from xai_sdk.chat import system, user
        print("✅ Grok SDK available")
    except ImportError:
        print("❌ Grok SDK not available. Install with: pip install xai-sdk")
        return

    # Get API key
    api_key = os.getenv('XAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("❌ Please set XAI_API_KEY environment variable")
        print("export XAI_API_KEY='your-actual-api-key'")
        return

    try:
        # Initialize client
        client = Client(api_key=api_key)
        print("✅ Grok client initialized")

        # Create chat session
        chat = client.chat.create(model="grok-4")

        # Set up the analyst role
        chat.append(system("""
        You are an expert financial analyst specializing in stock market analysis.
        Provide comprehensive analysis including:
        - Technical analysis
        - Market sentiment
        - Risk assessment
        - Trading recommendations
        - Price targets
        Be concise but thorough in your analysis.
        """))

        # Example stock data
        stock_data = """
        Stock: AAPL (Apple Inc.)
        Current Price: $175.50
        RSI: 68 (approaching overbought)
        MACD: 2.5 (bullish momentum)
        Volume: 1.2x average volume
        20 SMA: $172.00
        50 SMA: $168.00
        Recent News: Strong iPhone sales, AI integration announcements
        """

        # Request analysis
        chat.append(user(f"Please analyze this stock data and provide trading insights:\n{stock_data}"))

        # Get response
        response = chat.get_response()

        if response and response.content:
            print("\n🧠 GROK AI ANALYSIS:")
            print("=" * 50)
            print(response.content)
            print("=" * 50)
        else:
            print("❌ No response received")

    except Exception as e:
        print(f"❌ Error: {e}")

def example_news_sentiment_analysis():
    """Example of news sentiment analysis with Grok"""

    try:
        from xai_sdk import Client
        from xai_sdk.chat import system, user
    except ImportError:
        print("❌ Grok SDK not available")
        return

    api_key = os.getenv('XAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("❌ Please set XAI_API_KEY environment variable")
        return

    try:
        client = Client(api_key=api_key)
        chat = client.chat.create(model="grok-4")

        # Set up sentiment analysis role
        chat.append(system("""
        You are a financial news analyst. Analyze the sentiment of news headlines
        and provide structured output with:
        1. Overall sentiment (positive/negative/neutral)
        2. Confidence level (0-1)
        3. Key factors
        4. Market impact
        """))

        # Example news headlines
        news_headlines = [
            "Apple reports record quarterly earnings",
            "Tech sector faces regulatory scrutiny",
            "Federal Reserve signals potential rate cuts",
            "Market volatility increases amid geopolitical tensions",
            "AI stocks surge on breakthrough announcements"
        ]

        chat.append(user(f"Analyze the sentiment of these news headlines:\n" + "\n".join(news_headlines)))

        response = chat.get_response()

        if response and response.content:
            print("\n📰 NEWS SENTIMENT ANALYSIS:")
            print("=" * 50)
            print(response.content)
            print("=" * 50)
        else:
            print("❌ No sentiment analysis received")

    except Exception as e:
        print(f"❌ Error: {e}")

def example_structured_analysis():
    """Example of structured analysis with Pydantic models"""

    try:
        from xai_sdk import Client
        from xai_sdk.chat import system, user
        from pydantic import BaseModel, Field
        print("✅ Required libraries available")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        return

    api_key = os.getenv('XAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("❌ Please set XAI_API_KEY environment variable")
        return

    try:
        client = Client(api_key=api_key)
        chat = client.chat.create(model="grok-4")

        # Define structured output model
        class StockRecommendation(BaseModel):
            symbol: str = Field(description="Stock symbol")
            sentiment: str = Field(description="Market sentiment")
            confidence: float = Field(description="Confidence score", ge=0, le=1)
            recommendation: str = Field(description="Trading recommendation")
            target_price: float = Field(description="Target price")
            risk_level: str = Field(description="Risk level")

        # Set up structured analysis
        chat.append(system("""
        Provide stock analysis in a structured format that can be parsed into JSON.
        Include sentiment, confidence, recommendation, target price, and risk level.
        """))

        chat.append(user("Analyze TSLA (Tesla) stock with current price $250.00"))

        response = chat.get_response()

        if response and response.content:
            print("\n📊 STRUCTURED ANALYSIS:")
            print("=" * 50)
            print(response.content)
            print("=" * 50)

            # Try to parse structured response
            try:
                # This is a simplified example - in practice you'd use more robust parsing
                lines = response.content.split('\n')
                analysis = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        analysis[key.strip()] = value.strip()

                print("\n📋 PARSED ANALYSIS:")
                for key, value in analysis.items():
                    print(f"{key}: {value}")

            except Exception as parse_error:
                print(f"Note: Could not parse structured response: {parse_error}")
        else:
            print("❌ No structured analysis received")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all examples"""
    print("🚀 Grok API Integration Examples")
    print("=" * 60)

    print("\n1️⃣ Stock Analysis Example")
    example_grok_stock_analysis()

    print("\n2️⃣ News Sentiment Analysis Example")
    example_news_sentiment_analysis()

    print("\n3️⃣ Structured Analysis Example")
    example_structured_analysis()

    print("\n✅ Examples completed!")
    print("\nTo use these features in the bot:")
    print("1. Set your XAI_API_KEY environment variable")
    print("2. Run the bot: python backtest.py")
    print("3. Use /grok <symbol> command in Telegram")

if __name__ == "__main__":
    main()