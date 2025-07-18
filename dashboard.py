import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from backtest import get_historical_data, calculate_technical_indicators, calculate_supertrend, plot_oi_matplotlib, run_backtrader_backtest, train_prediction_model, calculate_confidence_score
import plotly.graph_objects as go
import yfinance as yf
import time
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
import json
from kiteconnect import KiteConnect
from alpaca_trade_api.rest import REST as AlpacaREST
from telegram import Bot
import requests
import config
import pysqlcipher3.dbapi2 as sqlcipher
import tempfile
import os
import jwt
from breeze_connect import BreezeConnect
import urllib.parse
# from icicidirect_api import ICICIDirect  # Uncomment if icicidirect-api is available
import plotly.express as px

# --- JWT Management ---
def get_jwt_secret():
    from dashboard import get_secret
    return get_secret('jwt_secret', 'changemejwt')

def issue_jwt(username, expires_in=3600):
    secret = get_jwt_secret()
    payload = {
        'sub': username,
        'exp': datetime.utcnow() + timedelta(seconds=expires_in)
    }
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_jwt(token):
    secret = get_jwt_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload['sub']
    except Exception:
        return None

# --- API Auth (API key or JWT) ---
def check_api_auth(x_api_key: str = None, authorization: str = None):
    api_key = get_api_key()
    if x_api_key == api_key:
        return True
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
        user = verify_jwt(token)
        if user:
            return True
    raise HTTPException(status_code=401, detail="Invalid API Key or JWT")

# --- REST API Integration (FastAPI) ---
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
import threading
from starlette.middleware.wsgi import WSGIMiddleware
from fpdf import FPDF

# --- API Key Management ---
API_KEY_HEADER = "X-API-Key"
def get_api_key():
    # Use the same get_secret as for other secrets
    from dashboard import get_secret
    return get_secret('unified_api_key', 'changeme')

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != get_api_key():
        raise HTTPException(status_code=401, detail="Invalid API Key")

api_app = FastAPI()

@api_app.post("/api/token")
def get_token(username: str = Query(...), password: str = Query(...)):
    # Only admin can issue tokens (for demo, check against config)
    from dashboard import get_secret
    valid_user = get_secret('USERNAME', 'admin')
    valid_pass = get_secret('PASSWORD', 'admin')
    if username == valid_user and password == valid_pass:
        token = issue_jwt(username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@api_app.get("/api/unified")
def unified_api(
    symbol: str = Query(...),
    expiry: str = Query(None),
    backend: str = Query('auto'),
    blackscholes: bool = Query(True),
    start_date: str = Query(None),
    end_date: str = Query(None),
    x_api_key: str = Header(None),
    authorization: str = Header(None)
):
    check_api_auth(x_api_key, authorization)
    from backtest import recommend_trades
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
            use_black_scholes=blackscholes,
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
    response = {
        'summary': summary_text,
        'oi_chart': oi_chart_file.name if result.get('oi_chart_saved') else None,
        'backtest_csv': backtest_csv_file.name if result.get('backtest_csv_saved') else None
    }
    return JSONResponse(response)

@api_app.get("/api/unified/oi_chart")
def oi_chart_api(path: str = Query(...), x_api_key: str = Header(None), authorization: str = Header(None)):
    check_api_auth(x_api_key, authorization)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/png", filename=os.path.basename(path))

@api_app.get("/api/unified/backtest_csv")
def backtest_csv_api(path: str = Query(...), x_api_key: str = Header(None), authorization: str = Header(None)):
    check_api_auth(x_api_key, authorization)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# Start FastAPI in a background thread
import uvicorn

def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

# --- PDF Export Utility (Enhanced) ---
def export_pdf(summary_text, oi_chart_path=None, backtest_csv_path=None, pdf_path=None):
    pdf = FPDF()
    pdf.add_page()
    # Add logo (placeholder)
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=30)
        pdf.ln(20)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Unified Trading Analytics Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    # Add summary
    pdf.set_font("Arial", size=11)
    for line in summary_text.splitlines():
        pdf.multi_cell(0, 8, line)
    # Add OI chart
    if oi_chart_path and os.path.exists(oi_chart_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Open Interest Chart", ln=True, align='C')
        pdf.image(oi_chart_path, x=10, y=20, w=180)
    # Add backtest table
    if backtest_csv_path and os.path.exists(backtest_csv_path):
        import pandas as pd
        df = pd.read_csv(backtest_csv_path)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Backtest Results", ln=True, align='C')
        pdf.set_font("Arial", size=9)
        col_width = pdf.w / (len(df.columns) + 1)
        row_height = 6
        # Table header
        for col in df.columns:
            pdf.cell(col_width, row_height, str(col), border=1)
        pdf.ln(row_height)
        # Table rows
        for i, row in df.iterrows():
            for col in df.columns:
                pdf.cell(col_width, row_height, str(row[col]), border=1)
            pdf.ln(row_height)
            if i > 30:
                pdf.cell(0, row_height, "... (truncated)", border=1)
                break
    if not pdf_path:
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
    pdf.output(pdf_path)
    return pdf_path

# --- Simple Authentication Layer ---
def check_login():
    import streamlit as st
    import config
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        # Use OAuth login
        oauth_login()
    if st.button('Logout'):
        user_data = {k: v for k, v in st.session_state.items() if isinstance(k, str) and not k.startswith('_')}
        save_user_data(st.session_state.get('username', getattr(config, 'USERNAME', 'admin')), user_data)
        st.session_state['logged_in'] = False
        st.experimental_rerun()

# --- Admin Secret Management UI (add JWT secret) ---
def admin_secret_manager():
    st.sidebar.write('## 🔒 Admin: Manage API & OAuth Secrets')
    st.sidebar.info('All secrets are stored encrypted in the local DB.')
    # Only show to admin
    if st.session_state.get('username', '') == getattr(config, 'USERNAME', 'admin'):
        with st.sidebar.expander('OAuth Client IDs & Secrets', expanded=False):
            google_client_id = st.text_input('Google OAuth Client ID', value=get_secret('google_client_id', ''), type='password')
            google_client_secret = st.text_input('Google OAuth Client Secret', value=get_secret('google_client_secret', ''), type='password')
            github_client_id = st.text_input('GitHub OAuth Client ID', value=get_secret('github_client_id', ''), type='password')
            github_client_secret = st.text_input('GitHub OAuth Client Secret', value=get_secret('github_client_secret', ''), type='password')
            if st.button('Save OAuth Secrets'):
                set_secret('google_client_id', google_client_id)
                set_secret('google_client_secret', google_client_secret)
                set_secret('github_client_id', github_client_id)
                set_secret('github_client_secret', github_client_secret)
                st.success('OAuth secrets saved!')
        with st.sidebar.expander('Other API Keys', expanded=False):
            openai_api_key = st.text_input('OpenAI API Key', value=get_secret('openai_api_key', ''), type='password')
            grok_api_key = st.text_input('Grok API Key', value=get_secret('grok_api_key', ''), type='password')
            gemini_api_key = st.text_input('Gemini API Key', value=get_secret('gemini_api_key', ''), type='password')
            huggingface_api_key = st.text_input('HuggingFace API Key', value=get_secret('huggingface_api_key', ''), type='password')
            anthropic_api_key = st.text_input('Anthropic API Key', value=get_secret('anthropic_api_key', ''), type='password')
            unified_api_key = st.text_input('Unified API Key (for REST API)', value=get_secret('unified_api_key', 'changeme'), type='password')
            jwt_secret = st.text_input('JWT Secret (for REST API)', value=get_secret('jwt_secret', 'changemejwt'), type='password')
            if st.button('Save API Keys'):
                set_secret('openai_api_key', openai_api_key)
                set_secret('grok_api_key', grok_api_key)
                set_secret('gemini_api_key', gemini_api_key)
                set_secret('huggingface_api_key', huggingface_api_key)
                set_secret('anthropic_api_key', anthropic_api_key)
                set_secret('unified_api_key', unified_api_key)
                set_secret('jwt_secret', jwt_secret)
                st.success('API keys saved!')

# --- OAuth Authentication (Google & GitHub) ---
import streamlit_authenticator as stauth

def oauth_login():
    # Load secrets from encrypted DB
    google_client_id = get_secret('google_client_id', '')
    google_client_secret = get_secret('google_client_secret', '')
    github_client_id = get_secret('github_client_id', '')
    github_client_secret = get_secret('github_client_secret', '')
    # Allow user to choose provider
    st.write('Login with:')
    provider = st.radio('OAuth Provider', ['Google', 'GitHub'], horizontal=True)
    # Prepare config for streamlit-authenticator
    credentials = {
        'oauth': {
            'google': {
                'client_id': google_client_id,
                'client_secret': google_client_secret,
            },
            'github': {
                'client_id': github_client_id,
                'client_secret': github_client_secret,
            },
        }
    }
    # Use streamlit-authenticator for OAuth
    authenticator = stauth.Authenticate(
        credentials,
        cookie_name='trading_app_auth',
        key='trading_app_auth_key',
        cookie_expiry_days=1
    )
    name, auth_status, username = authenticator.login('Login', 'main')
    if auth_status:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success(f'Logged in as {username}')
        # Load user data from DB
        user_data = load_user_data(username)
        for k, v in user_data.items():
            st.session_state[k] = v
        return True
    elif auth_status is False:
        st.error('Invalid login')
        st.stop()
    else:
        st.stop()

check_login()

st.title('Stock Analysis & Backtest Dashboard')

# Sidebar for navigation
page = st.sidebar.selectbox('Choose a page', ['Analysis & Backtest', 'Paper Trading', 'Parameter Optimization'])

if page == 'Analysis & Backtest':
    symbol = st.text_input('Stock Symbol (e.g. TCS.NS, AAPL)', 'TCS.NS')
    end_date = st.date_input('End Date', datetime.today())
    start_date = st.date_input('Start Date', datetime.today() - timedelta(days=365))
    # --- OI Feature Controls ---
    st.sidebar.write('---')
    st.sidebar.write('### OI Feature Options')
    include_oi_delta = st.sidebar.checkbox('Include Real-time OI Delta (slow)', value=False)
    oi_delta_interval = st.sidebar.number_input('OI Delta Poll Interval (sec)', min_value=10, max_value=300, value=60)
    expiry = st.sidebar.text_input('Option Expiry (for OI features, optional)', '')
    if include_oi_delta:
        st.sidebar.warning('Enabling OI delta will slow down analysis (waits for polling interval).')
    # --- Feature Selection ---
    default_features = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'Stoch_K', 'Stoch_D',
        'ATR', 'Volume_Ratio', 'Price_Change', 'Price_Change_5', 'Price_Change_10', 'Volatility',
        'PCR', 'Total_Call_OI', 'Total_Put_OI',
        'Hist_Call_OI_Mean', 'Hist_Put_OI_Mean', 'Hist_Call_OI_Std', 'Hist_Put_OI_Std',
        'Delta_Call_OI_Mean', 'Delta_Put_OI_Mean'
    ]
    feature_columns = st.multiselect('ML Feature Columns', default_features, default=default_features)
    run_analysis = st.button('Run Analysis')

    if run_analysis:
        st.write(f'Fetching data for {symbol}...')
        df = get_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df is None or df.empty:
            st.error('No data found!')
        else:
            df = calculate_technical_indicators(
                df,
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                expiry=expiry if expiry else None,
                poll_oi_delta=include_oi_delta,
                oi_delta_interval=oi_delta_interval
            )
            df = calculate_supertrend(df)
            st.write('## Technical Chart')
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            st.plotly_chart(fig)
            if symbol.endswith('.NS') or symbol in ['NIFTY', 'BANKNIFTY']:
                st.write('## Open Interest (OI) Chart')
                try:
                    from nsepython import option_chain
                    oc = option_chain(symbol.replace('.NS',''))
                    plot_oi_matplotlib(oc['filtered']['data'], symbol)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.warning(f'OI chart error: {e}')
            df['Signal'] = 0
            df['ML_Signal'] = 0
            df['Tech_Signal'] = 0
            df.loc[df['Close'] > df['SMA_20'], 'Signal'] = 1
            df.loc[df['Close'] < df['SMA_20'], 'Signal'] = -1
            df['ML_Signal'] = df['Signal']
            df['Tech_Signal'] = df['Signal']
            st.write('## Backtest Results')
            # Pass feature_columns to prepare_features (assume backtest.py will use it)
            run_backtrader_backtest(
                df,
                signal_col='Signal',
                commission=0.001,
                slippage=0.001,
                risk_per_trade=0.02,
                initial_cash=100000,
                extra_signal_cols=['ML_Signal', 'Tech_Signal']
            )
            oi_cols = [col for col in df.columns if 'OI' in col]
            if oi_cols:
                st.write('### OI Features (Snapshot, Historical, Delta)')
                st.dataframe(df[oi_cols].tail(10))

    st.write('---')
    st.write('### Advanced: Upload and Run Custom ML Script')
    st.warning('Security Warning: Uploaded code will be executed on the server. Only upload scripts you trust!')
    uploaded_script = st.file_uploader('Upload Python Script for Custom Signal (must define a function custom_signal(df) -> pd.Series)', type=['py'], key='custom_ml_script')
    custom_signal = None
    if uploaded_script:
        import types
        code = uploaded_script.read().decode('utf-8')
        local_vars = {}
        try:
            exec(code, {}, local_vars)
            if 'custom_signal' in local_vars:
                custom_signal = local_vars['custom_signal']
                st.success('Custom signal function loaded!')
            else:
                st.error('Script must define a function named custom_signal(df)')
        except Exception as e:
            st.error(f'Error in script: {e}')
    if custom_signal and run_analysis and df is not None and not df.empty:
        try:
            st.write('## Custom ML Signal Output')
            custom_signals = custom_signal(df)
            st.write(custom_signals)
            df['Custom_Signal'] = custom_signals
            st.write('## Backtest with Custom Signal')
            run_backtrader_backtest(
                df,
                signal_col='Custom_Signal',
                commission=0.001,
                slippage=0.001,
                risk_per_trade=0.02,
                initial_cash=100000
            )
        except Exception as e:
            st.error(f'Error running custom signal: {e}')

def send_notification(email, subject, message):
    # Simple SMTP notification (configure your SMTP server below)
    smtp_server = get_secret('smtp_server', 'smtp.example.com')
    smtp_port = int(get_secret('smtp_port', 587))
    smtp_user = get_secret('smtp_user', '')
    smtp_pass = get_secret('smtp_pass', '')
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [email], msg.as_string())
    except Exception as e:
        st.warning(f'Notification error: {e}')

def send_telegram_message(token, chat_id, message):
    try:
        bot = Bot(token=token)
        bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        st.warning(f'Telegram notification error: {e}')

def send_pushover_notification(user_key, api_token, message):
    try:
        resp = requests.post('https://api.pushover.net/1/messages.json', data={
            'token': api_token,
            'user': user_key,
            'message': message
        })
        if resp.status_code != 200:
            st.warning(f'Pushover error: {resp.text}')
    except Exception as e:
        st.warning(f'Pushover notification error: {e}')

# --- Broker Abstraction ---
class Broker:
    def place_order(self, symbol, side, size, price):
        raise NotImplementedError
    def get_positions(self):
        raise NotImplementedError
    def get_cash(self):
        raise NotImplementedError
    def get_trades(self):
        raise NotImplementedError

class DemoBroker(Broker):
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}
        self.trades = []
    def place_order(self, symbol, side, size, price):
        if side == 'buy' and self.cash >= size * price:
            self.cash -= size * price
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            self.trades.append({'symbol': symbol, 'side': side, 'size': size, 'price': price, 'time': datetime.now()})
        elif side == 'sell' and self.positions.get(symbol, 0) >= size:
            self.cash += size * price
            self.positions[symbol] -= size
            self.trades.append({'symbol': symbol, 'side': side, 'size': size, 'price': price, 'time': datetime.now()})
    def get_positions(self):
        return self.positions
    def get_cash(self):
        return self.cash
    def get_trades(self):
        return self.trades

class ZerodhaBroker(Broker):
    def __init__(self, api_key, api_secret, access_token=None):
        self.kite = KiteConnect(api_key=api_key)
        self.api_secret = api_secret
        self.access_token = access_token
        if access_token:
            self.kite.set_access_token(access_token)
        self.positions = {}
        self.trades = []
        self.cash = 100000  # Placeholder for demo; fetch from API in real use
    def login_url(self):
        return self.kite.login_url()
    def set_access_token(self, request_token):
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        self.access_token = data['access_token']
        self.kite.set_access_token(self.access_token)
    def place_order(self, symbol, side, size, price):
        # Real implementation: use self.kite.place_order(...)
        try:
            order_type = self.kite.ORDER_TYPE_MARKET
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if side == 'buy' else self.kite.TRANSACTION_TYPE_SELL
            self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=size,
                order_type=order_type,
                product=self.kite.PRODUCT_MIS
            )
            self.trades.append({'symbol': symbol, 'side': side, 'size': size, 'price': price, 'time': datetime.now()})
        except Exception as e:
            st.warning(f'Zerodha order error: {e}')
    def get_positions(self):
        try:
            pos = self.kite.positions()['net']
            return {p['tradingsymbol']: p['quantity'] for p in pos}
        except Exception as e:
            st.warning(f'Zerodha positions error: {e}')
            return {}
    def get_cash(self):
        try:
            margin = self.kite.margins()['equity']['available']['cash']
            return margin
        except Exception as e:
            st.warning(f'Zerodha cash error: {e}')
            return 0
    def get_trades(self):
        try:
            orders = self.kite.orders()
            return [{'symbol': o['tradingsymbol'], 'side': o['transaction_type'], 'size': o['quantity'], 'price': o['average_price'], 'time': o['order_timestamp']} for o in orders]
        except Exception as e:
            st.warning(f'Zerodha trades error: {e}')
            return []

class AlpacaBroker(Broker):
    def __init__(self, api_key, api_secret, base_url="https://paper-api.alpaca.markets"):
        self.api = AlpacaREST(api_key, api_secret, base_url)
    def place_order(self, symbol, side, size, price):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=side,
                type='market',
                time_in_force='day'
            )
        except Exception as e:
            st.warning(f'Alpaca order error: {e}')
    def get_positions(self):
        try:
            positions = self.api.list_positions()
            return {p.symbol: int(float(p.qty)) for p in positions}
        except Exception as e:
            st.warning(f'Alpaca positions error: {e}')
            return {}
    def get_cash(self):
        try:
            account = self.api.get_account()
            return float(account.cash)
        except Exception as e:
            st.warning(f'Alpaca cash error: {e}')
            return 0
    def get_trades(self):
        try:
            orders = self.api.list_orders(status='all', limit=100)
            return [{'symbol': o.symbol, 'side': o.side, 'size': o.qty, 'price': o.filled_avg_price, 'time': o.filled_at} for o in orders if o.filled_at]
        except Exception as e:
            st.warning(f'Alpaca trades error: {e}')
            return []

class BreezeBroker(Broker):
    def __init__(self, api_key, api_secret, session_token):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.breeze = BreezeConnect(api_key=api_key)
        self.authenticated = False
        try:
            self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
            self.authenticated = True
        except Exception as e:
            st.warning(f'BreezeConnect authentication error: {e}')
        self.positions = {}
        self.trades = []
        self.cash = 0
    def get_customer_details(self):
        try:
            return self.breeze.get_customer_details(api_session=self.session_token)
        except Exception as e:
            st.warning(f'Breeze customer details error: {e}')
            return {}
    def get_funds(self):
        try:
            return self.breeze.get_funds()
        except Exception as e:
            st.warning(f'Breeze funds error: {e}')
            return {}
    def get_demat_holdings(self):
        try:
            return self.breeze.get_demat_holdings()
        except Exception as e:
            st.warning(f'Breeze demat holdings error: {e}')
            return {}
    def get_positions(self):
        # For demo, return demat holdings as positions
        return self.get_demat_holdings()
    def get_cash(self):
        funds = self.get_funds()
        return float(funds.get('available_margin', 0)) if funds else 0
    def get_trades(self):
        # For demo, return order list for today
        try:
            today_iso = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()[:19] + '.000Z'
            orders = self.breeze.get_order_list(exchange_code="NSE", from_date=today_iso, to_date=today_iso)
            return orders
        except Exception as e:
            st.warning(f'Breeze order list error: {e}')
            return []
    def place_order(self, symbol, side, size, price, product="cash", order_type="limit", exchange="NSE"):
        try:
            action = 'buy' if side == 'buy' else 'sell'
            order = self.breeze.place_order(
                stock_code=symbol,
                exchange_code=exchange,
                product=product,
                action=action,
                order_type=order_type,
                quantity=str(size),
                price=str(price),
                validity="day"
            )
            self.trades.append(order)
        except Exception as e:
            st.warning(f'Breeze order error: {e}')
    def get_historical_data(self, symbol, interval, from_date, to_date, exchange="NSE", product_type="cash"):
        try:
            return self.breeze.get_historical_data(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=symbol,
                exchange_code=exchange,
                product_type=product_type
            )
        except Exception as e:
            st.warning(f'Breeze historical data error: {e}')
            return []

class GrowwBroker(Broker):
    def __init__(self):
        # As of 2024, Groww does not provide a public trading API for retail users.
        # This is a placeholder for future integration.
        st.warning('Groww API is not available for retail users as of 2024.')
    def place_order(self, symbol, side, size, price):
        st.warning('Groww API is not available.')
    def get_positions(self):
        return {}
    def get_cash(self):
        return 0
    def get_trades(self):
        return []

class ICICIBreezeBroker(Broker):
    def __init__(self, api_key, api_secret, session_token):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.breeze = BreezeConnect(api_key=api_key)
        self.login_url = "https://api.icicidirect.com/apiuser/login?api_key=" + urllib.parse.quote_plus(api_key)
        try:
            self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
        except Exception as e:
            st.warning(f'ICICI Breeze session error: {e}')
        self.positions = {}
        self.trades = []
        self.cash = 0
    def get_login_url(self):
        return self.login_url
    def get_customer_details(self):
        try:
            return self.breeze.get_customer_details(api_session=self.session_token)
        except Exception as e:
            st.warning(f'ICICI Breeze customer details error: {e}')
            return {}
    def get_funds(self):
        try:
            return self.breeze.get_funds()
        except Exception as e:
            st.warning(f'ICICI Breeze funds error: {e}')
            return {}
    def get_holdings(self):
        try:
            return self.breeze.get_demat_holdings()
        except Exception as e:
            st.warning(f'ICICI Breeze holdings error: {e}')
            return {}
    def get_historical(self, stock_code, exchange_code, product_type, from_date, to_date, interval="1day"):
        try:
            return self.breeze.get_historical_data(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=stock_code,
                exchange_code=exchange_code,
                product_type=product_type
            )
        except Exception as e:
            st.warning(f'ICICI Breeze historical error: {e}')
            return []
    def place_order(self, **kwargs):
        try:
            return self.breeze.place_order(**kwargs)
        except Exception as e:
            st.warning(f'ICICI Breeze order error: {e}')
            return {}
    def get_order_list(self, exchange_code, from_date, to_date):
        try:
            return self.breeze.get_order_list(
                exchange_code=exchange_code,
                from_date=from_date,
                to_date=to_date
            )
        except Exception as e:
            st.warning(f'ICICI Breeze order list error: {e}')
            return []

# --- Streamlit Sidebar for Broker Selection ---
st.sidebar.write('## Broker Settings')
broker_choice = st.sidebar.selectbox('Select Broker', ['Demo (Simulated)', 'Zerodha Kite', 'Breeze (ICICI)', 'Groww', 'Alpaca', 'Interactive Brokers', 'ICICI Breeze'])
if broker_choice == 'Demo (Simulated)':
    broker = DemoBroker()
    st.sidebar.success('Using Demo Broker (Simulated)')
elif broker_choice == 'Zerodha Kite':
    kite_key = st.sidebar.text_input('Kite API Key', value=get_secret('zerodha_api_key', ''), type='password')
    kite_secret = st.sidebar.text_input('Kite API Secret', value=get_secret('zerodha_api_secret', ''), type='password')
    access_token = st.sidebar.text_input('Kite Access Token (after login)', value=get_secret('zerodha_access_token', ''), type='password')
    broker = ZerodhaBroker(kite_key, kite_secret, access_token)
    if not access_token and kite_key and kite_secret:
        st.sidebar.write('1. Click the link below to login to Zerodha and get your request token:')
        login_url = broker.login_url()
        st.sidebar.markdown(f'[Zerodha Login]({login_url})')
        request_token = st.sidebar.text_input('Paste request token here')
        if st.sidebar.button('Generate Access Token') and request_token:
            try:
                broker.set_access_token(request_token)
                set_secret('zerodha_access_token', broker.access_token)
                st.sidebar.success('Access token set!')
            except Exception as e:
                st.sidebar.error(f'Error: {e}')
    st.sidebar.info('Zerodha integration: Use the login flow to get your access token.')
elif broker_choice == 'Breeze (ICICI)':
    breeze_api_key = st.sidebar.text_input('Breeze API Key', value=get_secret('breeze_api_key', ''), type='password')
    breeze_api_secret = st.sidebar.text_input('Breeze API Secret', value=get_secret('breeze_api_secret', ''), type='password')
    breeze_session_token = st.sidebar.text_input('Breeze Session Token (update daily)', value=get_secret('breeze_session_token', ''), type='password')
    broker = BreezeBroker(breeze_api_key, breeze_api_secret, breeze_session_token)
    if st.sidebar.button('Save Breeze Credentials'):
        set_secret('breeze_api_key', breeze_api_key)
        set_secret('breeze_api_secret', breeze_api_secret)
        set_secret('breeze_session_token', breeze_session_token)
        st.sidebar.success('Breeze credentials saved!')
    st.sidebar.info('Breeze (ICICI) integration: Enter your API key, secret, and daily session token. Generate session token daily from the Breeze portal.')
elif broker_choice == 'Groww':
    broker = GrowwBroker()
    st.sidebar.info('Groww API is not available for retail users as of 2024.')
elif broker_choice == 'Alpaca':
    alpaca_key = st.sidebar.text_input('Alpaca API Key', value=get_secret('alpaca_api_key', ''), type='password')
    alpaca_secret = st.sidebar.text_input('Alpaca API Secret', value=get_secret('alpaca_api_secret', ''), type='password')
    broker = AlpacaBroker(alpaca_key, alpaca_secret)
    st.sidebar.success('Using Alpaca Broker (US stocks, paper/live).')
elif broker_choice == 'Interactive Brokers':
    ib_user = st.sidebar.text_input('IB Username', value=get_secret('ib_user', ''))
    ib_pass = st.sidebar.text_input('IB Password', value=get_secret('ib_password', ''), type='password')
    broker = DemoBroker()  # For now, fallback to demo
    st.sidebar.info('Interactive Brokers integration coming soon!')
elif broker_choice == 'ICICI Breeze':
    icici_api_key = st.sidebar.text_input('ICICI API Key', value=get_secret('icici_breeze_api_key', ''), type='password')
    icici_api_secret = st.sidebar.text_input('ICICI API Secret', value=get_secret('icici_breeze_api_secret', ''), type='password')
    icici_session_token = st.sidebar.text_input('ICICI Session Token (daily)', value=get_secret('icici_breeze_session_token', ''), type='password')
    broker = ICICIBreezeBroker(icici_api_key, icici_api_secret, icici_session_token)
    st.sidebar.markdown(f'[Generate Session Token]({broker.get_login_url()})')
    st.sidebar.info('Visit the above link daily, log in, and paste the session token here.')
    if st.sidebar.button('Save ICICI Breeze Credentials'):
        set_secret('icici_breeze_api_key', icici_api_key)
        set_secret('icici_breeze_api_secret', icici_api_secret)
        set_secret('icici_breeze_session_token', icici_session_token)
        st.sidebar.success('ICICI Breeze credentials saved!')
    # --- Order Placement UI ---
    st.write('---')
    st.header('ICICI Breeze: Place Order')
    order_tab, fno_tab, options_tab, adv_tab = st.tabs(["Equity/Cash", "Futures", "Options", "Advanced"])
    with order_tab:
        st.subheader('Equity/Cash Order')
        eq_symbol = st.text_input('Stock Code (e.g., RELIANCE)', key='icici_eq_symbol')
        eq_exchange = st.selectbox('Exchange', ['NSE', 'BSE'], key='icici_eq_exchange')
        eq_action = st.selectbox('Action', ['buy', 'sell'], key='icici_eq_action')
        eq_order_type = st.selectbox('Order Type', ['market', 'limit', 'stop'], key='icici_eq_order_type')
        eq_qty = st.number_input('Quantity', min_value=1, value=1, key='icici_eq_qty')
        eq_price = st.number_input('Price (for limit/stop)', min_value=0.0, value=0.0, key='icici_eq_price')
        eq_validity = st.selectbox('Validity', ['day', 'ioc'], key='icici_eq_validity')
        if st.button('Place Equity Order'):
            try:
                order_kwargs = dict(
                    stock_code=eq_symbol,
                    exchange_code=eq_exchange,
                    product='cash',
                    action=eq_action,
                    order_type=eq_order_type,
                    quantity=str(eq_qty),
                    validity=eq_validity
                )
                if eq_order_type != 'market':
                    order_kwargs['price'] = str(eq_price)
                order_resp = broker.place_order(**order_kwargs)
                st.success(f'Order placed! Response: {order_resp}')
            except Exception as e:
                st.error(f'Order error: {e}')
    with fno_tab:
        st.subheader('Futures Order')
        fut_symbol = st.text_input('Futures Code (e.g., NIFTY)', key='icici_fut_symbol')
        fut_exchange = st.selectbox('Exchange', ['NFO'], key='icici_fut_exchange')
        fut_action = st.selectbox('Action', ['buy', 'sell'], key='icici_fut_action')
        fut_order_type = st.selectbox('Order Type', ['market', 'limit', 'stop'], key='icici_fut_order_type')
        fut_qty = st.number_input('Quantity', min_value=1, value=1, key='icici_fut_qty')
        fut_price = st.number_input('Price (for limit/stop)', min_value=0.0, value=0.0, key='icici_fut_price')
        fut_validity = st.selectbox('Validity', ['day', 'ioc'], key='icici_fut_validity')
        fut_expiry = st.text_input('Expiry Date (YYYY-MM-DDTHH:MM:SS.000Z)', key='icici_fut_expiry')
        if st.button('Place Futures Order'):
            try:
                order_kwargs = dict(
                    stock_code=fut_symbol,
                    exchange_code=fut_exchange,
                    product='futures',
                    action=fut_action,
                    order_type=fut_order_type,
                    quantity=str(fut_qty),
                    validity=fut_validity,
                    expiry_date=fut_expiry
                )
                if fut_order_type != 'market':
                    order_kwargs['price'] = str(fut_price)
                order_resp = broker.place_order(**order_kwargs)
                st.success(f'Futures order placed! Response: {order_resp}')
            except Exception as e:
                st.error(f'Futures order error: {e}')
    with options_tab:
        st.subheader('Options Order')
        opt_symbol = st.text_input('Options Code (e.g., NIFTY)', key='icici_opt_symbol')
        opt_exchange = st.selectbox('Exchange', ['NFO'], key='icici_opt_exchange')
        opt_action = st.selectbox('Action', ['buy', 'sell'], key='icici_opt_action')
        opt_order_type = st.selectbox('Order Type', ['market', 'limit', 'stop'], key='icici_opt_order_type')
        opt_qty = st.number_input('Quantity', min_value=1, value=1, key='icici_opt_qty')
        opt_price = st.number_input('Price (for limit/stop)', min_value=0.0, value=0.0, key='icici_opt_price')
        opt_validity = st.selectbox('Validity', ['day', 'ioc'], key='icici_opt_validity')
        opt_expiry = st.text_input('Expiry Date (YYYY-MM-DDTHH:MM:SS.000Z)', key='icici_opt_expiry')
        opt_strike = st.text_input('Strike Price', key='icici_opt_strike')
        opt_right = st.selectbox('Right', ['call', 'put'], key='icici_opt_right')
        if st.button('Place Options Order'):
            try:
                order_kwargs = dict(
                    stock_code=opt_symbol,
                    exchange_code=opt_exchange,
                    product='options',
                    action=opt_action,
                    order_type=opt_order_type,
                    quantity=str(opt_qty),
                    validity=opt_validity,
                    expiry_date=opt_expiry,
                    strike_price=opt_strike,
                    right=opt_right
                )
                if opt_order_type != 'market':
                    order_kwargs['price'] = str(opt_price)
                order_resp = broker.place_order(**order_kwargs)
                st.success(f'Options order placed! Response: {order_resp}')
            except Exception as e:
                st.error(f'Options order error: {e}')
    with adv_tab:
        st.subheader('Advanced Order Types (Cover, Bracket)')
        adv_symbol = st.text_input('Stock Code (e.g., RELIANCE)', key='icici_adv_symbol')
        adv_exchange = st.selectbox('Exchange', ['NSE', 'BSE', 'NFO'], key='icici_adv_exchange')
        adv_action = st.selectbox('Action', ['buy', 'sell'], key='icici_adv_action')
        adv_order_type = st.selectbox('Order Type', ['cover', 'bracket'], key='icici_adv_order_type')
        adv_qty = st.number_input('Quantity', min_value=1, value=1, key='icici_adv_qty')
        adv_price = st.number_input('Price', min_value=0.0, value=0.0, key='icici_adv_price')
        adv_validity = st.selectbox('Validity', ['day', 'ioc'], key='icici_adv_validity')
        adv_stop_loss = st.number_input('Stop Loss Price', min_value=0.0, value=0.0, key='icici_adv_stop_loss')
        adv_target = st.number_input('Target Price (Bracket)', min_value=0.0, value=0.0, key='icici_adv_target')
        adv_trailing_sl = st.number_input('Trailing Stop Loss (Bracket)', min_value=0.0, value=0.0, key='icici_adv_trailing_sl')
        if st.button('Place Advanced Order'):
            try:
                order_kwargs = dict(
                    stock_code=adv_symbol,
                    exchange_code=adv_exchange,
                    product='cash',
                    action=adv_action,
                    order_type=adv_order_type,
                    quantity=str(adv_qty),
                    validity=adv_validity,
                    price=str(adv_price)
                )
                if adv_order_type == 'cover':
                    order_kwargs['stop_loss'] = str(adv_stop_loss)
                if adv_order_type == 'bracket':
                    order_kwargs['stop_loss'] = str(adv_stop_loss)
                    order_kwargs['target'] = str(adv_target)
                    order_kwargs['trailing_sl'] = str(adv_trailing_sl)
                order_resp = broker.place_order(**order_kwargs)
                st.success(f'Advanced order placed! Response: {order_resp}')
            except Exception as e:
                st.error(f'Advanced order error: {e}')
                if 'admin_error_log' not in st.session_state:
                    st.session_state['admin_error_log'] = []
                st.session_state['admin_error_log'].append(f'Advanced order error: {e}')
    # --- Order Status Tracking & Cancellation ---
    st.write('---')
    st.header('Order Status & Cancellation')
    auto_refresh = st.checkbox('Auto-refresh Order List', value=False, key='icici_auto_refresh')
    refresh_interval = st.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2, key='icici_refresh_interval')
    last_refresh = st.session_state.get('icici_last_refresh', 0)
    now = time.time()
    if auto_refresh and now - last_refresh > refresh_interval:
        st.session_state['icici_last_refresh'] = now
        st.experimental_rerun()
    if st.button('Refresh Order List') or (auto_refresh and now - last_refresh > refresh_interval):
        try:
            today_iso = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()[:19] + '.000Z'
            orders = broker.get_order_list(exchange_code='NSE', from_date=today_iso, to_date=today_iso)
            prev_orders = st.session_state.get('icici_order_list', [])
            st.session_state['icici_order_list'] = orders
            # Detect status changes
            prev_status = {o.get('order_id'): o.get('status') for o in prev_orders if 'order_id' in o}
            for o in orders:
                oid = o.get('order_id')
                status = o.get('status')
                if oid and status and prev_status.get(oid) and prev_status[oid] != status:
                    st.success(f'Order {oid} status changed: {prev_status[oid]} → {status}')
                    # Send notification if enabled
                    if 'Telegram' in notify_channel and telegram_token and telegram_chat_id:
                        send_telegram_message(telegram_token, telegram_chat_id, f'Order {oid} status changed: {prev_status[oid]} → {status}')
                    if 'Email' in notify_channel and email_notify:
                        send_notification(email_notify, 'Order Status Update', f'Order {oid} status changed: {prev_status[oid]} → {status}')
                    if 'Pushover' in notify_channel and pushover_user_key and pushover_api_token:
                        send_pushover_notification(pushover_user_key, pushover_api_token, f'Order {oid} status changed: {prev_status[oid]} → {status}')
        except Exception as e:
            st.error(f'Order list error: {e}')
            if 'admin_error_log' not in st.session_state:
                st.session_state['admin_error_log'] = []
            st.session_state['admin_error_log'].append(f'Order list error: {e}')
    orders = st.session_state.get('icici_order_list', [])
    if orders:
        df_orders = pd.DataFrame(orders)
        st.dataframe(df_orders)
        cancel_id = st.text_input('Order ID to Cancel', key='icici_cancel_order_id')
        if st.button('Cancel Order') and cancel_id:
            try:
                cancel_resp = broker.breeze.cancel_order(order_id=cancel_id)
                st.success(f'Order cancelled! Response: {cancel_resp}')
            except Exception as e:
                st.error(f'Cancel order error: {e}')
                if 'admin_error_log' not in st.session_state:
                    st.session_state['admin_error_log'] = []
                st.session_state['admin_error_log'].append(f'Cancel order error: {e}')
    # --- Admin Error Log & Analytics ---
    if st.session_state.get('username', '') == getattr(config, 'USERNAME', 'admin'):
        st.write('---')
        st.subheader('Admin Error Log & Analytics')
        error_log = st.session_state.get('admin_error_log', [])
        if error_log:
            # Error table
            err_df = pd.DataFrame({'error': error_log})
            err_df['type'] = err_df['error'].str.extract(r'(\w+ error)')
            err_df['time'] = pd.Timestamp.now()
            st.dataframe(err_df.tail(20))
            # Error counts
            err_counts = err_df['type'].value_counts().reset_index()
            err_counts.columns = ['Error Type', 'Count']
            st.write('### Error Counts')
            st.dataframe(err_counts)
            # Error chart
            st.write('### Error Frequency')
            chart_df = err_df.groupby(['type', err_df['time'].dt.date]).size().reset_index(name='count')
            if not chart_df.empty:
                fig = px.bar(chart_df, x='time', y='count', color='type', barmode='group', title='Error Frequency by Type')
                st.plotly_chart(fig)
            # Export error log
            st.download_button('Download Error Log (TXT)', '\n'.join(error_log), file_name='icici_admin_error_log.txt')
        else:
            st.info('No recent errors.')
    # --- Real-time Streaming UI ---
    st.sidebar.write('---')
    st.sidebar.write('### Real-time Streaming (Websockets)')
    stock_token = st.sidebar.text_input('Stock Token (e.g., 4.1!2885 for RELIANCE NSE)')
    stream_interval = st.sidebar.selectbox('Interval', ['1minute', '1day'], index=0)
    if 'breeze_streaming' not in st.session_state:
        st.session_state['breeze_streaming'] = False
    if st.sidebar.button('Start Streaming') and stock_token:
        st.session_state['breeze_streaming'] = True
        st.session_state['breeze_ticks'] = []
        def on_ticks(ticks):
            if 'breeze_ticks' not in st.session_state:
                st.session_state['breeze_ticks'] = []
            st.session_state['breeze_ticks'].append(ticks)
        try:
            broker.breeze.ws_connect()
            broker.breeze.on_ticks = on_ticks
            broker.breeze.subscribe_feeds(stock_token=stock_token, interval=stream_interval)
            st.sidebar.success('Streaming started!')
        except Exception as e:
            st.sidebar.error(f'Error starting streaming: {e}')
    if st.sidebar.button('Stop Streaming'):
        st.session_state['breeze_streaming'] = False
        try:
            broker.breeze.unsubscribe_feeds(stock_token=stock_token, interval=stream_interval)
            broker.breeze.ws_disconnect()
            st.sidebar.success('Streaming stopped!')
        except Exception as e:
            st.sidebar.error(f'Error stopping streaming: {e}')
    if st.session_state.get('breeze_streaming', False):
        st.write('## Live Ticks (ICICI Breeze)')
        ticks = st.session_state.get('breeze_ticks', [])
        if ticks:
            st.dataframe(pd.DataFrame(ticks))
        else:
            st.info('Waiting for live ticks...')

# --- Streamlit Sidebar for Notification Settings ---
st.sidebar.write('## Notification Settings')
notify_channel = st.sidebar.multiselect('Notification Channel(s)', ['Email', 'Telegram', 'Pushover'], default=['Email'])
telegram_token = st.sidebar.text_input('Telegram Bot Token', value=get_secret('telegram_bot_token', ''), type='password')
telegram_chat_id = st.sidebar.text_input('Telegram Chat ID', value=get_secret('telegram_chat_id', ''))
pushover_user_key = st.sidebar.text_input('Pushover User Key', value=get_secret('pushover_user_key', ''))
pushover_api_token = st.sidebar.text_input('Pushover API Token', value=get_secret('pushover_api_token', ''), type='password')

# SMTP settings (optional, for email notifications)
smtp_server = st.sidebar.text_input('SMTP Server', value=get_secret('smtp_server', 'smtp.example.com'))
smtp_port = st.sidebar.number_input('SMTP Port', value=int(get_secret('smtp_port', 587)), min_value=1, max_value=65535)
smtp_user = st.sidebar.text_input('SMTP Username', value=get_secret('smtp_user', ''))
smtp_pass = st.sidebar.text_input('SMTP Password', value=get_secret('smtp_pass', ''), type='password')

# --- Unified Recommendation Section ---
if 'Unified Recommendation' not in st.session_state:
    st.session_state['Unified Recommendation'] = False

if st.sidebar.button('Unified Recommendation'):
    st.session_state['Unified Recommendation'] = not st.session_state['Unified Recommendation']

if st.session_state['Unified Recommendation']:
    st.header('Unified Recommendation & Analytics')
    symbol = st.text_input('Symbol', 'NIFTY', key='unified_symbol')
    expiry = st.text_input('Option Expiry (e.g., 17-Jul-2025)', '', key='unified_expiry')
    r = st.number_input('Risk-free Rate (r)', value=0.06)
    sigma = st.number_input('Volatility (sigma)', value=0.2)
    start_date = st.date_input('Start Date', datetime.today() - timedelta(days=365), key='unified_start')
    end_date = st.date_input('End Date', datetime.today(), key='unified_end')
    backend = st.radio('Technical Indicator Backend', options=['auto', 'talib', 'pandas'], index=0, help='Use TA-Lib if available, otherwise pandas/numpy')
    use_black_scholes = st.checkbox('Include Black-Scholes Option Analytics', value=True)
    # --- OI Feature Controls for Unified ---
    st.write('---')
    st.write('### OI Feature Options')
    include_oi_delta = st.checkbox('Include Real-time OI Delta (slow)', value=False, key='unified_oi_delta')
    oi_delta_interval = st.number_input('OI Delta Poll Interval (sec)', min_value=10, max_value=300, value=60, key='unified_oi_interval')
    if include_oi_delta:
        st.warning('Enabling OI delta will slow down analysis (waits for polling interval).')
    # Add feature selection for unified as well
    st.write('---')
    st.write('### ML Feature Selection')
    feature_columns_unified = st.multiselect('ML Feature Columns (Unified)', default_features, default=default_features, key='unified_features')
    run_unified = st.button('Run Unified Recommendation')
    summary_text = ''
    oi_chart_path = None
    backtest_csv_path = None
    pdf_path = None
    if run_unified:
        from backtest import recommend_trades
        st.write('---')
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
                expiry=expiry if expiry else None,
                r=r,
                sigma=sigma,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                backend=backend,
                use_black_scholes=use_black_scholes,
                poll_oi_delta=include_oi_delta,
                oi_delta_interval=oi_delta_interval,
                feature_columns=feature_columns_unified,
                export_oi_chart_path=oi_chart_file.name,
                export_backtest_csv_path=backtest_csv_file.name
            )
        except Exception as e:
            print(f'Error: {e}')
            result = {}
        sys.stdout = sys_stdout
        summary_text = buffer.getvalue()
        st.code(summary_text, language='text')
        st.session_state['unified_summary_text'] = summary_text
        oi_chart_path = oi_chart_file.name if result.get('oi_chart_saved') else None
        backtest_csv_path = backtest_csv_file.name if result.get('backtest_csv_saved') else None
        pdf_path = export_pdf(summary_text, oi_chart_path, backtest_csv_path)
        if 'oi_features' in result and isinstance(result['oi_features'], dict):
            st.write('### OI Features (Snapshot, Historical, Delta)')
            st.dataframe(pd.DataFrame([result['oi_features']]))
    if st.session_state.get('unified_summary_text'):
        st.download_button('Download Summary as TXT', st.session_state['unified_summary_text'], file_name=f'{symbol}_unified_summary.txt')
    if oi_chart_path and os.path.exists(oi_chart_path):
        with open(oi_chart_path, 'rb') as f:
            st.download_button('Download OI Chart (PNG)', f, file_name=f'{symbol}_oi_chart.png')
    if backtest_csv_path and os.path.exists(backtest_csv_path):
        with open(backtest_csv_path, 'rb') as f:
            st.download_button('Download Backtest Results (CSV)', f, file_name=f'{symbol}_backtest.csv')
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as f:
            st.download_button('Download Full Report (PDF)', f, file_name=f'{symbol}_unified_report.pdf')

if page == 'Paper Trading':
    st.header('Paper Trading Simulator')
    username = st.text_input('Username (for multi-user support)', 'user1', key='user_name')
    email_notify = st.text_input('Notification Email (optional)', '', key='notify_email')
    symbols = st.text_input('Stock Symbol(s) for Paper Trading (comma separated)', 'TCS.NS', key='paper_symbol')
    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
    account_size = st.number_input('Account Size', min_value=1000, value=100000, step=1000)
    risk_per_trade = st.number_input('Risk per Trade (%)', min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
    commission = st.number_input('Commission per Trade (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    interval = st.selectbox('Price Update Interval (seconds)', [5, 10, 30, 60], index=1)
    strategy = st.selectbox('Signal Strategy', ['SMA20', 'ML', 'Advanced ML', 'Ensemble', 'Custom Script'], index=0)
    trailing_stop_pct = st.number_input('Trailing Stop (%)', min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100
    export_trades = st.button('Export Trades as CSV')
    auto_refresh = st.checkbox('Auto-refresh (every interval)', value=False)
    # User model/strategy upload
    st.write('### Upload Your ML Model (.pkl/.joblib) or Strategy Script (.py)')
    uploaded_model = st.file_uploader('Upload ML Model', type=['pkl', 'joblib'])
    uploaded_script = st.file_uploader('Upload Python Strategy Script', type=['py'])
    if uploaded_model:
        st.session_state['user_model_file'] = uploaded_model.getvalue()
        st.success('Model uploaded!')
    if uploaded_script:
        st.session_state['user_script_file'] = uploaded_script.getvalue()
        st.success('Strategy script uploaded!')
    # Save/load user settings/portfolio
    st.write('### Save/Load Settings & Portfolio')
    if st.button('Save Settings & Portfolio'):
        user_data = {
            'username': username,
            'symbols': symbols,
            'account_size': account_size,
            'risk_per_trade': risk_per_trade,
            'commission': commission,
            'strategy': strategy,
            'positions': st.session_state.get(f'paper_{username}_positions', {}),
            'cash': st.session_state.get(f'paper_{username}_cash', account_size),
            'trades': st.session_state.get(f'paper_{username}_trades', []),
        }
        json_data = json.dumps(user_data)
        st.download_button('Download Settings', json_data, file_name=f'{username}_settings.json')
    uploaded_settings = st.file_uploader('Load Settings (JSON)', type=['json'])
    if uploaded_settings:
        user_data = json.loads(uploaded_settings.getvalue())
        st.session_state[f'paper_{username}_positions'] = user_data.get('positions', {})
        st.session_state[f'paper_{username}_cash'] = user_data.get('cash', account_size)
        st.session_state[f'paper_{username}_trades'] = user_data.get('trades', [])
        st.success('Settings loaded!')
    user_key = f'paper_{username}'
    if export_trades and user_key+'_trades' in st.session_state and st.session_state[user_key+'_trades']:
        trades_df = pd.DataFrame(st.session_state[user_key+'_trades'])
        st.download_button('Download Trades CSV', trades_df.to_csv(index=False), file_name=f'{username}_paper_trades.csv')
    if user_key+'_running' not in st.session_state:
        st.session_state[user_key+'_running'] = False
    if 'start_paper' in st.session_state and st.session_state['start_paper']:
        st.session_state[user_key+'_running'] = True
        st.session_state[user_key+'_positions'] = {sym: [] for sym in symbol_list}
        st.session_state[user_key+'_cash'] = account_size
        st.session_state[user_key+'_trades'] = []
        st.session_state[user_key+'_pnl'] = []
    if 'stop_paper' in st.session_state and st.session_state['stop_paper']:
        st.session_state[user_key+'_running'] = False
    if st.session_state[user_key+'_running']:
        st.success(f'Paper trading is running for {username}!')
        # --- Performance Analytics ---
        if 'portfolio_history' not in st.session_state:
            st.session_state['portfolio_history'] = []
        def get_last_price(sym):
            df = yf.download(sym, period='1d', interval='1m')
            if df is not None and not df.empty and 'Close' in df.columns:
                return df['Close'].iloc[-1]
            return 0
        portfolio_value = broker.get_cash() + sum([broker.get_positions().get(sym, 0) * get_last_price(sym) for sym in symbol_list])
        st.session_state['portfolio_history'].append({'time': datetime.now(), 'value': portfolio_value})
        total_pnl = 0
        total_trades = 0
        wins = 0
        losses = 0
        all_pnl = []
        for symbol in symbol_list:
            price_data = yf.download(symbol, period='1d', interval='1m')
            if price_data is not None and not price_data.empty:
                close_series = price_data['Close']
                if not isinstance(close_series, pd.Series):
                    close_series = pd.Series(close_series)
                last_price = close_series.iloc[-1]
                st.write(f'[{symbol}] Current Price: {last_price:.2f}')
                # Signal logic
                if strategy == 'SMA20':
                    sma20_series = close_series.rolling(window=20).mean()
                    if not isinstance(sma20_series, pd.Series):
                        sma20_series = pd.Series(sma20_series)
                    sma20 = sma20_series.iloc[-1]
                    signal = 0
                    if last_price > sma20:
                        signal = 1
                    elif last_price < sma20:
                        signal = -1
                elif strategy == 'ML':
                    hist_df = get_historical_data(symbol, (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d'))
                    if hist_df is not None and not hist_df.empty:
                        hist_df = calculate_technical_indicators(hist_df, symbol)
                        model, scaler, _ = train_prediction_model(hist_df)
                        conf, rec = calculate_confidence_score(hist_df, model, scaler)
                        signal = 1 if rec in ['STRONG BUY', 'BUY'] else -1 if rec in ['STRONG SELL', 'SELL'] else 0
                    else:
                        signal = 0
                elif strategy == 'Advanced ML':
                    # Placeholder for user-uploaded or advanced ML model
                    st.info('Advanced ML: Please upload your model or configure advanced ML here (future feature).')
                    signal = 0
                elif strategy == 'Ensemble':
                    votes = []
                    sma20_series = close_series.rolling(window=20).mean()
                    if not isinstance(sma20_series, pd.Series):
                        sma20_series = pd.Series(sma20_series)
                    sma20 = sma20_series.iloc[-1]
                    if last_price > sma20:
                        votes.append(1)
                    elif last_price < sma20:
                        votes.append(-1)
                    else:
                        votes.append(0)
                    hist_df = get_historical_data(symbol, (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d'))
                    if hist_df is not None and not hist_df.empty:
                        hist_df = calculate_technical_indicators(hist_df, symbol)
                        model, scaler, _ = train_prediction_model(hist_df)
                        conf, rec = calculate_confidence_score(hist_df, model, scaler)
                        ml_vote = 1 if rec in ['STRONG BUY', 'BUY'] else -1 if rec in ['STRONG SELL', 'SELL'] else 0
                        votes.append(ml_vote)
                    signal = int(round(sum(votes) / len(votes)))
                elif strategy == 'Custom Script':
                    # Placeholder for custom script execution
                    st.info('Custom Script: Please upload your Python strategy script to execute.')
                    signal = 0
                st.write(f'[{symbol}] Signal: {"BUY" if signal==1 else "SELL" if signal==-1 else "HOLD"}')
                # Use broker for order placement and position management
                positions = broker.get_positions()
                cash = broker.get_cash()
                trades = broker.get_trades()
                size = int((cash * risk_per_trade) / (last_price * 0.02)) if last_price > 0 else 0
                # Trailing stop logic (for demo broker, update positions dict)
                # NOTE: DemoBroker positions is a dict, so we cannot use .remove. For a real implementation, track entry prices and stops per position.
                # for pos in positions[:]:
                #     if 'trailing_stop' not in pos:
                #         pos['trailing_stop'] = pos['price'] * (1 - trailing_stop_pct)
                #     if last_price > pos['price']:
                #         pos['trailing_stop'] = max(pos['trailing_stop'], last_price * (1 - trailing_stop_pct))
                #     if last_price < pos['trailing_stop']:
                #         cash += last_price * pos['size'] * (1 - commission)
                #         trade_pnl = (last_price - pos['price']) * pos['size']
                #         trades.append({'symbol': symbol, 'action': 'stop', 'price': last_price, 'size': pos['size'], 'time': datetime.now(), 'pnl': trade_pnl})
                #         all_pnl.append(trade_pnl)
                #         total_pnl += trade_pnl
                #         total_trades += 1
                #         if trade_pnl > 0:
                #             wins += 1
                #         else:
                #             losses += 1
                #         positions.remove(pos)
                if signal == 1 and cash > last_price * size:
                    broker.place_order(symbol, 'buy', size, last_price)
                    if email_notify and 'Email' in notify_channel:
                        send_notification(email_notify, f'Paper Trade BUY {symbol}', f'Bought {size} of {symbol} at {last_price:.2f}')
                    if telegram_token and telegram_chat_id and 'Telegram' in notify_channel:
                        send_telegram_message(telegram_token, telegram_chat_id, f'Paper Trade BUY {symbol} at {last_price:.2f}')
                    if pushover_user_key and pushover_api_token and 'Pushover' in notify_channel:
                        send_pushover_notification(pushover_user_key, pushover_api_token, f'Paper Trade BUY {symbol} at {last_price:.2f}')
                elif signal == -1 and positions.get(symbol, 0) > 0:
                    broker.place_order(symbol, 'sell', positions[symbol], last_price)
                    trade_pnl = 0  # For demo, you can calculate PnL if you track entry price
                    if email_notify and 'Email' in notify_channel:
                        send_notification(email_notify, f'Paper Trade SELL {symbol}', f'Sold {positions[symbol]} of {symbol} at {last_price:.2f} (PnL: {trade_pnl:.2f})')
                    if telegram_token and telegram_chat_id and 'Telegram' in notify_channel:
                        send_telegram_message(telegram_token, telegram_chat_id, f'Paper Trade SELL {symbol} at {last_price:.2f} (PnL: {trade_pnl:.2f})')
                    if pushover_user_key and pushover_api_token and 'Pushover' in notify_channel:
                        send_pushover_notification(pushover_user_key, pushover_api_token, f'Paper Trade SELL {symbol} at {last_price:.2f} (PnL: {trade_pnl:.2f})')
                # For demo, update session state for display only
                st.session_state[user_key+'_positions'][symbol] = positions
                st.session_state[user_key+'_cash'] = cash
                st.session_state[user_key+'_trades'] = trades
        if total_trades > 0:
            win_rate = wins / total_trades
            sharpe = (pd.Series(all_pnl).mean() / (pd.Series(all_pnl).std() + 1e-9)) * (252 ** 0.5) if len(all_pnl) > 1 else 0
            max_dd = (pd.Series(all_pnl).cumsum().cummax() - pd.Series(all_pnl).cumsum()).max() if len(all_pnl) > 1 else 0
            st.write(f'Total P&L: {total_pnl:.2f}')
            st.write(f'Win Rate: {win_rate:.1%}')
            st.write(f'Sharpe Ratio: {sharpe:.2f}')
            st.write(f'Max Drawdown: {max_dd:.2f}')
        st.write(f'Cash: {st.session_state[user_key+"_cash"]:.2f}')
        st.write(f'Open Positions: {st.session_state[user_key+"_positions"]}')
        st.write(f'Trades: {st.session_state[user_key+"_trades"][-5:]}')
        # --- Analytics Section ---
        st.write('## Performance Analytics')
        # Equity curve
        hist_df = pd.DataFrame(st.session_state['portfolio_history'])
        if not hist_df.empty:
            import plotly.express as px
            fig = px.line(hist_df, x='time', y='value', title='Equity Curve')
            st.plotly_chart(fig)
            # Drawdown
            hist_df['cummax'] = hist_df['value'].cummax()
            hist_df['drawdown'] = hist_df['value'] - hist_df['cummax']
            fig_dd = px.area(hist_df, x='time', y='drawdown', title='Drawdown')
            st.plotly_chart(fig_dd)
        # Trade log
        st.write('## Trade Log')
        trades_raw = broker.get_trades() if broker.get_trades() is not None else []
        trades_df = pd.DataFrame(trades_raw)
        if not trades_df.empty:
            st.dataframe(trades_df)
        # Summary stats
        st.write('## Summary Statistics')
        if not trades_df.empty:
            pnl = trades_df['pnl'] if 'pnl' in trades_df.columns else pd.Series([0]*len(trades_df))
            win_rate = (pnl > 0).mean() if not pnl.empty else 0
            avg_trade = pnl.mean() if not pnl.empty else 0
            sharpe = (pnl.mean() / (pnl.std() + 1e-9)) * (252 ** 0.5) if len(pnl) > 1 else 0
            st.write(f'Win Rate: {win_rate:.1%}')
            st.write(f'Average Trade: {avg_trade:.2f}')
            st.write(f'Sharpe Ratio: {sharpe:.2f}')
        # Only rerun if Streamlit supports it
        if auto_refresh and hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()

if page == 'Parameter Optimization':
    st.header('Parameter Optimization')
    symbol = st.text_input('Stock Symbol for Optimization', 'TCS.NS', key='opt_symbol')
    sma_min = st.number_input('SMA Window Min', min_value=5, value=10, step=1)
    sma_max = st.number_input('SMA Window Max', min_value=5, value=50, step=1)
    sma_step = st.number_input('SMA Window Step', min_value=1, value=5, step=1)
    risk_min = st.number_input('Risk per Trade Min (%)', min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100
    risk_max = st.number_input('Risk per Trade Max (%)', min_value=0.1, max_value=10.0, value=5.0, step=0.1) / 100
    risk_step = st.number_input('Risk per Trade Step (%)', min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100
    # --- OI Feature Controls for Optimization ---
    st.write('---')
    st.write('### OI Feature Options')
    include_oi_delta_opt = st.checkbox('Include Real-time OI Delta (slow)', value=False, key='opt_oi_delta')
    oi_delta_interval_opt = st.number_input('OI Delta Poll Interval (sec)', min_value=10, max_value=300, value=60, key='opt_oi_interval')
    expiry_opt = st.text_input('Option Expiry (for OI features, optional)', '', key='opt_expiry')
    if include_oi_delta_opt:
        st.warning('Enabling OI delta will slow down optimization (waits for polling interval).')
    # --- Feature Selection for Optimization ---
    feature_columns_opt = st.multiselect('ML Feature Columns (Optimization)', default_features, default=default_features, key='opt_features')
    run_opt = st.button('Run Parameter Optimization')
    if run_opt:
        st.write('Running optimization...')
        results = []
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = get_historical_data(symbol, start_date, end_date)
        if df is None or df.empty:
            st.error('No data found!')
        else:
            for sma in range(int(sma_min), int(sma_max)+1, int(sma_step)):
                for risk in [risk_min + i*risk_step for i in range(int((risk_max-risk_min)/risk_step)+1)]:
                    df = calculate_technical_indicators(
                        df,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        expiry=expiry_opt if expiry_opt else None,
                        poll_oi_delta=include_oi_delta_opt,
                        oi_delta_interval=oi_delta_interval_opt
                    )
                    df['Signal'] = 0
                    df.loc[df['Close'] > df['Close'].rolling(window=sma).mean(), 'Signal'] = 1
                    df.loc[df['Close'] < df['Close'].rolling(window=sma).mean(), 'Signal'] = -1
                    # Use feature_columns_opt in ML pipeline (assume backtest.py will use it)
                    trades = df['Signal'].diff().fillna(0).abs().sum()
                    pnl = (df['Signal'].shift(1) * df['Close'].pct_change()).cumsum().iloc[-1]
                    win_rate = (df['Signal'].diff().fillna(0) > 0).mean()
                    sharpe = (df['Signal'].shift(1) * df['Close'].pct_change()).mean() / ((df['Signal'].shift(1) * df['Close'].pct_change()).std() + 1e-9) * (252 ** 0.5)
                    results.append({'SMA': sma, 'Risk': risk, 'Sharpe': sharpe, 'Win Rate': win_rate, 'P&L': pnl, 'Trades': trades})
            results_df = pd.DataFrame(results)
            st.write('## Optimization Results')
            st.dataframe(results_df)
            import plotly.express as px
            fig = px.scatter(results_df, x='SMA', y='Sharpe', color='Risk', size='P&L', hover_data=['Win Rate', 'Trades'])
            st.plotly_chart(fig)
            st.write('## Select Optimal Parameters')
            selected = st.selectbox('Select Row', results_df.index)
            if selected is not None:
                st.write(f'Optimal SMA: {results_df.loc[selected, "SMA"]}, Risk: {results_df.loc[selected, "Risk"]}')

# --- SQLite Encrypted DB for Multi-user Persistence ---
DB_PATH = 'user_data_encrypted.db'
DB_KEY = getattr(config, 'DB_ENCRYPTION_KEY', 'mysecretkey')

def get_db():
    conn = sqlcipher.connect(DB_PATH)
    conn.execute(f"PRAGMA key='{DB_KEY}';")
    # User data table
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        data TEXT
    )''')
    # Secrets table for API keys, OAuth client IDs/secrets, etc.
    conn.execute('''CREATE TABLE IF NOT EXISTS secrets (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    return conn

def load_user_data(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT data FROM users WHERE username=?', (username,))
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        import json
        return json.loads(row[0])
    return {}

def save_user_data(username, data):
    conn = get_db()
    cur = conn.cursor()
    import json
    cur.execute('REPLACE INTO users (username, data) VALUES (?, ?)', (username, json.dumps(data)))
    conn.commit()
    conn.close()

# --- Secrets Management ---
def set_secret(key, value):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('REPLACE INTO secrets (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()

def get_secret(key, default=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT value FROM secrets WHERE key=?', (key,))
    row = cur.fetchone()
    conn.close()
    if row and row[0] is not None:
        return row[0]
    return default

# Document deployment instructions at the bottom of the file
if __name__ == '__main__':
    st.write('''
    ## Deployment Instructions
    - To deploy on Streamlit Cloud:
      1. Push your code to GitHub.
      2. Go to https://streamlit.io/cloud and connect your repo.
      3. Set the main file to `stock_mf-predict-algo/dashboard.py`.
      4. Add any required secrets (API keys, DB_ENCRYPTION_KEY) in the Streamlit Cloud UI.
    - To deploy on Heroku:
      1. Add a `Procfile` with `web: streamlit run stock_mf-predict-algo/dashboard.py`.
      2. Add `requirements.txt` and push to Heroku.
    - To deploy on AWS:
      1. Use Elastic Beanstalk, EC2, or ECS with Docker.
      2. Expose port 8501 and run `streamlit run stock_mf-predict-algo/dashboard.py`.
    ''')