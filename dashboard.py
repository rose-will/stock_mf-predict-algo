import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from backtest import get_historical_data, calculate_technical_indicators, calculate_supertrend, plot_oi_matplotlib, run_backtrader_backtest
import plotly.graph_objects as go

st.title('Stock Analysis & Backtest Dashboard')

symbol = st.text_input('Stock Symbol (e.g. TCS.NS, AAPL)', 'TCS.NS')
end_date = st.date_input('End Date', datetime.today())
start_date = st.date_input('Start Date', datetime.today() - timedelta(days=365))
run_analysis = st.button('Run Analysis')

if run_analysis:
    st.write(f'Fetching data for {symbol}...')
    df = get_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is None or df.empty:
        st.error('No data found!')
    else:
        st.success('Data loaded!')
        df = calculate_technical_indicators(df, symbol)
        df = calculate_supertrend(df)
        st.write('## Technical Chart')
        # Plotly candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        st.plotly_chart(fig)
        # OI chart for Indian stocks
        if symbol.endswith('.NS') or symbol in ['NIFTY', 'BANKNIFTY']:
            st.write('## Open Interest (OI) Chart')
            try:
                from nsepython import option_chain
                oc = option_chain(symbol.replace('.NS',''))
                plot_oi_matplotlib(oc['filtered']['data'], symbol)
                st.pyplot(plt)
            except Exception as e:
                st.warning(f'OI chart error: {e}')
        # Example: generate dummy signals for demo
        df['Signal'] = 0
        df['ML_Signal'] = 0
        df['Tech_Signal'] = 0
        df.loc[df['Close'] > df['SMA_20'], 'Signal'] = 1
        df.loc[df['Close'] < df['SMA_20'], 'Signal'] = -1
        df['ML_Signal'] = df['Signal']
        df['Tech_Signal'] = df['Signal']
        st.write('## Backtest Results')
        run_backtrader_backtest(
            df,
            signal_col='Signal',
            commission=0.001,
            slippage=0.001,
            risk_per_trade=0.02,
            initial_cash=100000,
            extra_signal_cols=['ML_Signal', 'Tech_Signal']
        )