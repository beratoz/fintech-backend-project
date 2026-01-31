import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import time
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Fintech Real-Time Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Connection ---
# Use environment variable for DB URL or fallback to localhost for local testing outside docker
DB_URL = os.environ.get("DB_URL", "postgresql://user:password@localhost:5432/fintech")

@st.cache_resource
def get_engine():
    try:
        return create_engine(DB_URL)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

engine = get_engine()

# --- Auto Refresh Logic ---
# Streamlit re-runs the script from top to bottom on every interaction.
# To simulate "real-time", we use a loop with sleep and rerun, or a dedicated refresher component.
# For MVP simplicity, we'll use a refresh button or a simple sleep/rerun loop if "Auto Refresh" is checked.

# --- Sidebar ---
st.sidebar.title("Configuration ⚙️")

# Refresh Settings
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 2, 60, 5)

st.sidebar.markdown("---")

# Ticker Selection
# We fetch unique tickers from DB to populate the dropdown
if engine:
    try:
        tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM trade_signals", engine)
        available_tickers = tickers_df['ticker'].tolist()
        if not available_tickers:
             available_tickers = ["WAITING_FOR_DATA"]
    except Exception as e:
        st.sidebar.error("Error fetching tickers. Database might be initializing.")
        available_tickers = ["ERROR"]
else:
    available_tickers = ["NO_DB_CONNECTION"]

selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard connected to `fintech-postgres`.")

# --- Main Content ---
st.title(f"🚀 {selected_ticker} Analysis")

if selected_ticker and selected_ticker not in ["WAITING_FOR_DATA", "ERROR", "NO_DB_CONNECTION"] and engine:
    
    # Fetch Data
    query = f"""
        SELECT * FROM trade_signals 
        WHERE ticker = '{selected_ticker}' 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            # Preprocess
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df_sorted = df.sort_values('timestamp')

            # --- KPI Metrics (Top Row) ---
            latest = df_sorted.iloc[-1]
            
            # Extract JSON indicators safely
            try:
                if isinstance(latest['indicators'], str):
                    indicators = json.loads(latest['indicators'])
                else:
                    indicators = latest['indicators'] # Already dict if driver handled jsonb
            except:
                indicators = {}

            col1, col2, col3, col4 = st.columns(4)
            
            # Price
            col1.metric("Current Price", f"${latest['price']:.2f}")
            
            # Signal
            signal_map = {1: "BUY 🟢", 0: "HOLD ⚪"}
            signal_val = latest['prediction']
            col2.metric("Latest Signal", signal_map.get(signal_val, "UNKNOWN"))
            
            # Technicals
            rsi = indicators.get('RSI', 0)
            sentiment = indicators.get('Sentiment', 0)
            col3.metric("RSI (14)", f"{rsi:.2f}")
            col4.metric("Sentiment Score", f"{sentiment:.2f}")
            
            st.markdown("---")

            # --- Charts (Middle Row) ---
            # Create interactive Plotly chart
            fig = go.Figure()

            # Price Line
            fig.add_trace(go.Scatter(
                x=df_sorted['timestamp'], 
                y=df_sorted['price'],
                mode='lines',
                name='Price',
                line=dict(color='#2962FF', width=2)
            ))

            # Buy Signals Overlay
            # Filter where prediction == 1
            buys = df_sorted[df_sorted['prediction'] == 1]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['timestamp'], 
                    y=buys['price'],
                    mode='markers',
                    name='BUY Signal',
                    marker=dict(color='#00E676', size=12, symbol='circle', line=dict(color='white', width=1))
                ))

            fig.update_layout(
                title=f"{selected_ticker} Price Action & Signals",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # --- Raw Data (Bottom Row) ---
            with st.expander("View Recent Raw Data"):
                st.dataframe(df.head(10))

        else:
            st.warning(f"No data found for {selected_ticker} yet. Adjust filter or wait for stream.")
            
    except Exception as e:
        st.error(f"Error executing query: {e}")

else:
    if selected_ticker == "WAITING_FOR_DATA":
         st.info("Waiting for first batch of data from Spark Processor... (approx 1-2 mins on fresh start)")
    elif selected_ticker == "ERROR":
         st.error("Could not fetch tickers. Check database connection.")


# --- Rerun Logic ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
