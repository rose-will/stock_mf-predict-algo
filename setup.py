#!/usr/bin/env python3
"""
Setup script for Stock Analysis Bot
Helps users configure API keys and dependencies
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_config_file():
    """Create config.py from template"""
    if os.path.exists("config.py"):
        print("⚠️ config.py already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            return True

    if not os.path.exists("config_template.py"):
        print("❌ config_template.py not found")
        return False

    try:
        with open("config_template.py", "r") as template:
            content = template.read()

        with open("config.py", "w") as config:
            config.write(content)

        print("✅ config.py created from template")
        print("📝 Please edit config.py and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create config.py: {e}")
        return False

def check_git_ignore():
    """Check if .gitignore includes config.py"""
    if not os.path.exists(".gitignore"):
        print("⚠️ .gitignore not found")
        return False

    try:
        with open(".gitignore", "r") as f:
            content = f.read()

        if "config.py" in content:
            print("✅ config.py is in .gitignore")
            return True
        else:
            print("⚠️ config.py is not in .gitignore")
            return False
    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Stock Analysis Bot Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        return

    # Install dependencies
    if not install_dependencies():
        return

    # Create config file
    if not create_config_file():
        return

    # Check git ignore
    check_git_ignore()

    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit config.py and add your API keys")
    print("2. Get API keys from:")
    print("   - Telegram Bot: @BotFather")
    print("   - Google Gemini: https://makersuite.google.com/app/apikey")
    print("   - Hugging Face: https://huggingface.co/settings/tokens")
    print("   - Anthropic: https://console.anthropic.com/")
    print("3. Run the bot: python backtest.py")
    print("\n📚 For detailed instructions, see README.md")

if __name__ == "__main__":
    main()