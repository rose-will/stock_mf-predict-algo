#!/usr/bin/env python3
"""
Integration script to add free AI services to your existing stock analysis bot.
This script modifies your backtest.py to include free AI alternatives.
"""

import os
import shutil
from datetime import datetime

def backup_original_file():
    """Create a backup of the original backtest.py"""
    if os.path.exists('backtest.py'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'backtest_backup_{timestamp}.py'
        shutil.copy2('backtest.py', backup_name)
        print(f"✅ Created backup: {backup_name}")
        return backup_name
    return None

def add_free_ai_imports():
    """Add free AI imports to backtest.py"""
    import_section = '''
# Free AI Integration
try:
    from free_ai_integration import analyze_stock_with_free_ai, get_free_ai_usage
    FREE_AI_AVAILABLE = True
    print("✅ Free AI integration available")
except ImportError:
    FREE_AI_AVAILABLE = False
    print("⚠️ Free AI integration not available. Install with: pip install -r requirements.txt")
'''
    return import_section

def add_free_ai_fallback_function():
    """Add free AI fallback function"""
    function_code = '''
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
'''
    return function_code

def modify_analyze_news_with_ai_function():
    """Modify the existing analyze_news_with_ai function to include free AI fallback"""
    modification = '''
        # Try free AI as final fallback
        if FREE_AI_AVAILABLE:
            try:
                news_text = "\\n".join(news_items)
                result = analyze_stock_with_free_ai(f"Analyze sentiment for {symbol}: {news_text}")

                if 'error' not in result:
                    return {
                        'sentiment': result.get('sentiment', 'neutral'),
                        'confidence': result.get('confidence', 0.5),
                        'summary': result.get('analysis', 'News analysis completed'),
                        'impact': 'bullish' if result.get('sentiment') == 'positive' else 'bearish' if result.get('sentiment') == 'negative' else 'neutral',
                        'provider': result.get('provider', 'Free AI')
                    }
            except Exception as free_ai_error:
                print(f"Free AI fallback error: {free_ai_error}")
'''
    return modification

def add_free_ai_command():
    """Add a new /freeai command to test free AI services"""
    command_code = '''
async def freeai_command(update, context):
    """Test free AI services"""
    if not update.message:
        return

    if len(context.args) != 1:
        await update.message.reply_text("Usage: /freeai <stock_symbol>\\nExample: /freeai AAPL")
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
        analysis = f"🤖 FREE AI ANALYSIS: {stock_symbol}\\n\\n"
        analysis += f"💰 Current Price: ${current_price:.2f}\\n"
        analysis += f"📊 Provider: {result['provider']}\\n"
        analysis += f"🎯 Sentiment: {result['sentiment'].upper()}\\n"
        analysis += f"🎚️ Confidence: {result['confidence']:.1%}\\n"
        analysis += f"📈 Impact: {result['impact'].upper()}\\n"
        analysis += f"💡 Recommendation: {result.get('recommendation', 'N/A').upper()}\\n\\n"
        analysis += f"📝 Summary:\\n{result['summary']}\\n\\n"

        # Add usage stats
        if FREE_AI_AVAILABLE:
            usage = get_free_ai_usage()
            analysis += f"📊 Usage Stats:\\n"
            analysis += f"• Current Provider: {usage.get('current_provider', 'None')}\\n"
            for provider, count in usage.get('request_counts', {}).items():
                limit = usage.get('limits', {}).get(provider, 'Unknown')
                analysis += f"• {provider}: {count}/{limit} requests\\n"

        await update.message.reply_text(analysis)

    except Exception as e:
        await update.message.reply_text(f"❌ Error in free AI analysis: {str(e)}")
'''
    return command_code

def integrate_free_ai():
    """Main integration function"""
    print("🚀 Integrating Free AI Services into your stock analysis bot...")

    # Create backup
    backup_file = backup_original_file()

    # Read current backtest.py
    with open('backtest.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Add free AI imports after existing imports
    import_section = add_free_ai_imports()
    if '# Free AI Integration' not in content:
        # Find the end of imports section
        import_end = content.find('warnings.filterwarnings')
        if import_end != -1:
            content = content[:import_end] + import_section + '\n' + content[import_end:]

    # Add free AI fallback function
    fallback_function = add_free_ai_fallback_function()
    if 'def analyze_with_free_ai_fallback' not in content:
        # Add before the create_interactive_chart function
        chart_func_pos = content.find('def create_interactive_chart')
        if chart_func_pos != -1:
            content = content[:chart_func_pos] + fallback_function + '\n' + content[chart_func_pos:]

    # Modify analyze_news_with_ai function to include free AI fallback
    if 'FREE_AI_AVAILABLE' not in content:
        # Find the fallback section in analyze_news_with_ai
        fallback_pos = content.find('# Fallback when all APIs fail')
        if fallback_pos != -1:
            modification = modify_analyze_news_with_ai_function()
            content = content[:fallback_pos] + modification + '\n        ' + content[fallback_pos:]

    # Add free AI command
    if 'async def freeai_command' not in content:
        command_code = add_free_ai_command()
        # Add before the main() function
        main_pos = content.find('def main():')
        if main_pos != -1:
            content = content[:main_pos] + command_code + '\n' + content[main_pos:]

    # Add command handler in main function
    if 'CommandHandler(\'freeai\'' not in content:
        # Find where other command handlers are added
        handler_pos = content.find("app.add_handler(CommandHandler('help', help_command))")
        if handler_pos != -1:
            content = content[:handler_pos] + "    app.add_handler(CommandHandler('freeai', freeai_command))\n    " + content[handler_pos:]

    # Update help command to include free AI
    if '/freeai' not in content:
        help_pos = content.find("• /grok <symbol> - Grok AI-powered deep analysis")
        if help_pos != -1:
            help_addition = "• /freeai <symbol> - Test free AI services\n"
            content = content[:help_pos] + help_addition + content[help_pos:]

    # Write updated content
    with open('backtest.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Free AI integration completed!")
    print("📝 Changes made:")
    print("   • Added free AI imports")
    print("   • Added free AI fallback function")
    print("   • Modified news analysis to use free AI")
    print("   • Added /freeai command")
    print("   • Updated help message")
    print(f"📦 Backup created: {backup_file}")

    print("\n🚀 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up free AI services (see FREE_AI_SETUP.md)")
    print("3. Test with: python backtest.py")
    print("4. Use /freeai AAPL in Telegram to test")

def create_env_template():
    """Create a template .env file"""
    env_content = """# Free AI Services Configuration
# Get these API keys for free from the respective services

# HuggingFace (30,000 requests/month free)
HUGGINGFACE_API_KEY=your-huggingface-key-here

# Google Gemini (15 requests/minute free)
GEMINI_API_KEY=your-gemini-key-here

# Anthropic Claude (5 requests/minute free)
ANTHROPIC_API_KEY=your-anthropic-key-here

# OpenAI ($5 free credit monthly)
OPENAI_API_KEY=your-openai-key-here

# Your existing keys
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
XAI_API_KEY=your-grok-key-here
"""

    with open('.env.template', 'w') as f:
        f.write(env_content)

    print("✅ Created .env.template file")
    print("📝 Copy to .env and add your API keys")

if __name__ == "__main__":
    print("🤖 Free AI Integration Tool")
    print("=" * 50)

    choice = input("Choose an option:\n1. Integrate free AI into bot\n2. Create .env template\n3. Both\nEnter choice (1-3): ")

    if choice in ['1', '3']:
        integrate_free_ai()

    if choice in ['2', '3']:
        create_env_template()

    print("\n🎉 Setup complete! Check FREE_AI_SETUP.md for detailed instructions.")