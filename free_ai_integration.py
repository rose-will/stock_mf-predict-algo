#!/usr/bin/env python3
"""
Free AI Integration Module for Stock Analysis Bot
Provides multiple free AI services as alternatives to paid APIs
"""

import os
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# Import configuration
try:
    from config import *
    print("✅ Free AI config loaded from config.py")
except ImportError:
    print("⚠️ config.py not found. Using environment variables and defaults")
    # Fallback values
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

# Configuration for free AI services
FREE_AI_CONFIG = {
    'huggingface': {
        'enabled': True,
        'api_key': HUGGINGFACE_API_KEY,
        'models': ['microsoft/DialoGPT-medium', 'gpt2'],
        'requests_per_month': 30000
    },
    'ollama': {
        'enabled': True,  # Enabled for local LLMs
        'url': OLLAMA_BASE_URL,
        'models': [OLLAMA_MODEL, 'llama2', 'mistral', 'codellama'],
        'requests_per_month': float('inf')  # Unlimited
    },
    'openai_free': {
        'enabled': True,
        'api_key': OPENAI_API_KEY,
        'model': 'gpt-3.5-turbo',
        'requests_per_month': 1000  # Approximate
    },
    'anthropic': {
        'enabled': True,
        'api_key': ANTHROPIC_API_KEY,
        'model': 'claude-3-haiku-20240307',
        'requests_per_month': 43200  # 5 per minute * 24 * 30
    },
    'gemini': {
        'enabled': True,
        'api_key': GEMINI_API_KEY,
        'model': 'gemini-2.0-flash',
        'requests_per_month': 64800  # 15 per minute * 24 * 30
    },
    'deepseek': {
        'enabled': False,  # Disabled due to insufficient balance
        'api_key': DEEPSEEK_API_KEY,
        'model': 'deepseek-chat',
        'base_url': 'https://api.deepseek.com',
        'requests_per_month': 1000  # Approximate free tier limit
    }
}

class FreeAIAnalyzer:
    """Free AI Analysis Service with multiple providers"""

    def __init__(self):
        self.request_counts = {provider: 0 for provider in FREE_AI_CONFIG.keys()}
        self.last_reset = datetime.now()
        self.current_provider = None

    def reset_monthly_counts(self):
        """Reset monthly request counts"""
        now = datetime.now()
        if (now - self.last_reset).days >= 30:
            self.request_counts = {provider: 0 for provider in FREE_AI_CONFIG.keys()}
            self.last_reset = now

    def get_available_provider(self) -> Optional[str]:
        """Get the best available AI provider"""
        self.reset_monthly_counts()

        # Priority order: Ollama > Gemini > HuggingFace > Anthropic > OpenAI > DeepSeek
        providers = ['ollama', 'gemini', 'huggingface', 'anthropic', 'openai_free', 'deepseek']

        for provider in providers:
            config = FREE_AI_CONFIG[provider]
            if (config['enabled'] and
                self.request_counts[provider] < config['requests_per_month']):
                return provider

        return None

    def analyze_with_huggingface(self, text: str, model: str = "microsoft/DialoGPT-medium") -> Dict[str, Any]:
        """Analyze using Hugging Face Inference API"""
        try:
            api_key = FREE_AI_CONFIG['huggingface']['api_key']
            if not api_key:
                return {'error': 'HuggingFace API key not configured'}

            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}

            # Format prompt for stock analysis
            prompt = f"Analyze this stock data and provide sentiment (positive/negative/neutral), confidence (0-1), and recommendation (buy/hold/sell): {text}"

            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # Parse the response (HuggingFace responses vary by model)
                if isinstance(result, list) and len(result) > 0:
                    analysis_text = result[0].get('generated_text', str(result[0]))
                else:
                    analysis_text = str(result)

                return {
                    'provider': 'HuggingFace',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': f'HuggingFace API error: {response.status_code}'}

        except Exception as e:
            return {'error': f'HuggingFace error: {str(e)}'}

    def analyze_with_ollama(self, text: str, model: str = "llama2") -> Dict[str, Any]:
        """Analyze using local Ollama"""
        try:
            url = FREE_AI_CONFIG['ollama']['url']

            # Check if Ollama is running
            try:
                requests.get(f"{url}/api/tags", timeout=5)
            except:
                return {'error': 'Ollama not running. Start with: ollama serve'}

            prompt = f"""
            You are a financial analyst. Analyze this stock data and provide:
            1. Sentiment: [positive/negative/neutral]
            2. Confidence: [0-1 score]
            3. Recommendation: [buy/hold/sell]
            4. Brief analysis

            Stock data: {text}
            """

            response = requests.post(f'{url}/api/generate',
                                   json={
                                       "model": model,
                                       "prompt": prompt,
                                       "stream": False
                                   }, timeout=60)

            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get('response', '')

                return {
                    'provider': 'Ollama (Local)',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': f'Ollama API error: {response.status_code}'}

        except Exception as e:
            return {'error': f'Ollama error: {str(e)}'}

    def analyze_with_openai_free(self, text: str) -> Dict[str, Any]:
        """Analyze using OpenAI free tier"""
        try:
            api_key = FREE_AI_CONFIG['openai_free']['api_key']
            if not api_key:
                return {'error': 'OpenAI API key not configured'}

            # Try to import OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return {'error': 'OpenAI library not installed'}

            client = OpenAI(api_key=api_key)

            prompt = f"""
            Analyze this stock data and provide:
            1. Sentiment: [positive/negative/neutral]
            2. Confidence: [0-1 score]
            3. Recommendation: [buy/hold/sell]
            4. Brief analysis

            Stock data: {text}
            """

            response = client.chat.completions.create(
                model=FREE_AI_CONFIG['openai_free']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            if response.choices and response.choices[0].message:
                analysis_text = response.choices[0].message.content

                return {
                    'provider': 'OpenAI Free',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': 'No response from OpenAI'}

        except Exception as e:
            return {'error': f'OpenAI error: {str(e)}'}

    def analyze_with_anthropic(self, text: str) -> Dict[str, Any]:
        """Analyze using Anthropic Claude free tier"""
        try:
            api_key = FREE_AI_CONFIG['anthropic']['api_key']
            if not api_key:
                return {'error': 'Anthropic API key not configured'}

            # Try to import Anthropic
            try:
                import anthropic
            except ImportError:
                return {'error': 'Anthropic library not installed'}

            client = anthropic.Anthropic(api_key=api_key)

            prompt = f"""
            Analyze this stock data and provide:
            1. Sentiment: [positive/negative/neutral]
            2. Confidence: [0-1 score]
            3. Recommendation: [buy/hold/sell]
            4. Brief analysis

            Stock data: {text}
            """

            message = client.messages.create(
                model=FREE_AI_CONFIG['anthropic']['model'],
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            if message.content:
                analysis_text = message.content[0].text

                return {
                    'provider': 'Anthropic Claude',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': 'No response from Anthropic'}

        except Exception as e:
            return {'error': f'Anthropic error: {str(e)}'}

    def analyze_with_gemini(self, text: str) -> Dict[str, Any]:
        """Analyze using Google Gemini REST API"""
        try:
            api_key = FREE_AI_CONFIG['gemini']['api_key']
            model = FREE_AI_CONFIG['gemini']['model']
            if not api_key:
                return {'error': 'Gemini API key not configured'}

            # Use REST API directly
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': api_key
            }

            prompt = f"""
            Analyze this stock data and provide:
            1. Sentiment: [positive/negative/neutral]
            2. Confidence: [0-1 score]
            3. Recommendation: [buy/hold/sell]
            4. Brief analysis

            Stock data: {text}
            """

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()

                # Extract the response text
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        analysis_text = candidate['content']['parts'][0].get('text', '')
                    else:
                        analysis_text = str(candidate)
                else:
                    analysis_text = str(result)

                return {
                    'provider': 'Google Gemini',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': f'Gemini API error: {response.status_code} - {response.text}'}

        except Exception as e:
            return {'error': f'Gemini error: {str(e)}'}

    def analyze_with_deepseek(self, text: str) -> Dict[str, Any]:
        """Analyze using DeepSeek API with OpenAI-compatible format"""
        try:
            api_key = FREE_AI_CONFIG['deepseek']['api_key']
            base_url = FREE_AI_CONFIG['deepseek']['base_url']
            if not api_key:
                return {'error': 'DeepSeek API key not configured'}

            # Try to import OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                return {'error': 'OpenAI library not installed'}

            client = OpenAI(api_key=api_key, base_url=base_url)

            prompt = f"""
            Analyze this stock data and provide:
            1. Sentiment: [positive/negative/neutral]
            2. Confidence: [0-1 score]
            3. Recommendation: [buy/hold/sell]
            4. Brief analysis

            Stock data: {text}
            """

            response = client.chat.completions.create(
                model=FREE_AI_CONFIG['deepseek']['model'],
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                stream=False
            )

            if response.choices and response.choices[0].message:
                analysis_text = response.choices[0].message.content

                return {
                    'provider': 'DeepSeek',
                    'analysis': analysis_text,
                    'sentiment': self._extract_sentiment(analysis_text),
                    'confidence': self._extract_confidence(analysis_text),
                    'recommendation': self._extract_recommendation(analysis_text)
                }
            else:
                return {'error': 'No response from DeepSeek'}

        except Exception as e:
            return {'error': f'DeepSeek error: {str(e)}'}

    def analyze_stock_data(self, stock_data: str) -> Dict[str, Any]:
        """Main analysis function that tries multiple providers"""
        provider = self.get_available_provider()

        if not provider:
            return {
                'error': 'No free AI providers available',
                'provider': 'None',
                'analysis': 'All free AI services have reached their limits'
            }

        self.current_provider = provider
        self.request_counts[provider] += 1

        # Try the selected provider
        if provider == 'huggingface':
            return self.analyze_with_huggingface(stock_data)
        elif provider == 'ollama':
            return self.analyze_with_ollama(stock_data)
        elif provider == 'openai_free':
            return self.analyze_with_openai_free(stock_data)
        elif provider == 'anthropic':
            return self.analyze_with_anthropic(stock_data)
        elif provider == 'gemini':
            return self.analyze_with_gemini(stock_data)
        elif provider == 'deepseek':
            return self.analyze_with_deepseek(stock_data)
        else:
            return {'error': f'Unknown provider: {provider}'}

    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from AI response"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['positive', 'bullish', 'buy', 'strong']):
            return 'positive'
        elif any(word in text_lower for word in ['negative', 'bearish', 'sell', 'weak']):
            return 'negative'
        else:
            return 'neutral'

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from AI response"""
        try:
            # Look for confidence scores in the text
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', text.lower())
            if confidence_match:
                return float(confidence_match.group(1))

            # Estimate based on language
            if any(word in text.lower() for word in ['high', 'strong', 'very']):
                return 0.8
            elif any(word in text.lower() for word in ['medium', 'moderate']):
                return 0.6
            else:
                return 0.5
        except:
            return 0.5

    def _extract_recommendation(self, text: str) -> str:
        """Extract trading recommendation from AI response"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['buy', 'strong buy', 'purchase']):
            return 'buy'
        elif any(word in text_lower for word in ['sell', 'strong sell', 'short']):
            return 'sell'
        else:
            return 'hold'

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            'current_provider': self.current_provider,
            'request_counts': self.request_counts,
            'limits': {provider: config['requests_per_month']
                      for provider, config in FREE_AI_CONFIG.items()}
        }

# Global instance
free_ai_analyzer = FreeAIAnalyzer()

def analyze_stock_with_free_ai(stock_data: str) -> Dict[str, Any]:
    """Convenience function for stock analysis"""
    return free_ai_analyzer.analyze_stock_data(stock_data)

def get_free_ai_usage() -> Dict[str, Any]:
    """Get usage statistics"""
    return free_ai_analyzer.get_usage_stats()

# List of available AI providers/models (for user selection)
AVAILABLE_AI_MODELS = [
    'ollama',
    'gemini',
    'huggingface',
    'anthropic',
    'openai_free',
    'deepseek',
    'grok'  # If you want to include Grok as an option
]

def get_available_ai_models() -> list:
    """Return the list of available AI models/providers."""
    return AVAILABLE_AI_MODELS

# Example usage
if __name__ == "__main__":
    # Test the free AI integration
    test_data = """
    Stock: AAPL
    Current Price: $175.50
    RSI: 68
    MACD: 2.5
    Volume: 1.2x average
    """

    print("🧪 Testing Free AI Integration...")
    result = analyze_stock_with_free_ai(test_data)
    print(f"Provider: {result.get('provider', 'Unknown')}")
    print(f"Analysis: {result.get('analysis', 'No analysis')}")
    print(f"Sentiment: {result.get('sentiment', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0)}")
    print(f"Recommendation: {result.get('recommendation', 'Unknown')}")

    print("\n📊 Usage Stats:")
    stats = get_free_ai_usage()
    for provider, count in stats['request_counts'].items():
        limit = stats['limits'][provider]
        print(f"{provider}: {count}/{limit} requests")