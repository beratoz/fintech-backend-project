import time
import json
import logging
import random
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Constants
KAFKA_BROKER = "kafka:29092"
KAFKA_TOPIC = "stock-prices"
TICKERS = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "AAPL", "GOOGL"]
MACRO_TICKERS = ["^TNX", "DX-Y.NYB"]
SLEEP_INTERVAL = 60

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_kafka_producer():
    """Attempts to create a Kafka Producer with retry logic."""
    producer = None
    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            logger.info("Successfully connected to Kafka.")
        except NoBrokersAvailable:
            logger.warning("Kafka not ready. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error connecting to Kafka: {e}")
            time.sleep(5)
    return producer

def fetch_macro_state():
    """
    Fetches latest close price for Macro Indicators.
    Returns a dict: {"^TNX": 4.10, "DX-Y.NYB": 103.5}
    """
    state = {}
    for ticker in MACRO_TICKERS:
        try:
            # Fetch last 2 days to be safe and get latest close
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                last_price = data["Close"].iloc[-1]
                state[ticker] = float(last_price)
        except Exception as e:
            logger.error(f"Error fetching macro {ticker}: {e}")
            state[ticker] = 0.0 # Fallback
    return state

def fetch_history(ticker_symbol):
    """Fetches 1 year of historical data for a ticker."""
    logger.info(f"Fetching history for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="5y")
        
        records = []
        for index, row in history.iterrows():
            record = {
                "ticker": ticker_symbol,
                "timestamp": int(index.timestamp() * 1000),
                "price": float(row["Close"]),
                "data_type": "HISTORY",
                # Synthetic/Mock enrichment for history (since point-in-time macro is hard to sync perfectly here without complex logic)
                "tnx_chg": 0.0,
                "dxy_chg": 0.0,
                "sentiment_score": 0.0
            }
            records.append(record)
        return records
    except Exception as e:
        logger.error(f"Error fetching history for {ticker_symbol}: {e}")
        return []

def fetch_live_price(ticker_symbol, macro_state, prev_macro_state):
    """
    Fetches the latest live price for a ticker and enriches with Macro changes.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Using 'fast_info' or fetching 1 day period to get latest
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            latest = data.iloc[-1]
            
            # Calculate Macro Changes
            tnx_now = macro_state.get("^TNX", 0)
            tnx_prev = prev_macro_state.get("^TNX", tnx_now)
            tnx_chg = tnx_now - tnx_prev

            dxy_now = macro_state.get("DX-Y.NYB", 0)
            dxy_prev = prev_macro_state.get("DX-Y.NYB", dxy_now)
            dxy_chg = dxy_now - dxy_prev

            # Mock Sentiment (Gaussian)
            sentiment_score = random.gauss(0, 0.5)

            return {
                "ticker": ticker_symbol,
                "timestamp": int(latest.name.timestamp() * 1000),
                "price": float(latest["Close"]),
                "data_type": "LIVE",
                "tnx_chg": float(tnx_chg),
                "dxy_chg": float(dxy_chg),
                "sentiment_score": float(sentiment_score)
            }
    except Exception as e:
        logger.error(f"Error fetching live data for {ticker_symbol}: {e}")
    return None

def main():
    producer = create_kafka_producer()

    # PHASE 1: Bootstrapping (History)
    logger.info("Starting Phase 1: Bootstrapping History")
    for ticker in TICKERS:
        history_data = fetch_history(ticker)
        for record in history_data:
            producer.send(KAFKA_TOPIC, value=record)
        logger.info(f"Sent {len(history_data)} historical records for {ticker}")
        producer.flush()

    logger.info("Phase 1 Complete. Transitioning to Phase 2: Live Streaming")

    # Initialize Macro State
    prev_macro_state = fetch_macro_state()
    logger.info(f"Initial Macro State: {prev_macro_state}")

    # PHASE 2: Live Streaming
    while True:
        logger.info("Fetching live data cycle...")
        
        # Update Macro State
        current_macro_state = fetch_macro_state()
        logger.debug(f"Current Macro State: {current_macro_state}")

        for ticker in TICKERS:
            record = fetch_live_price(ticker, current_macro_state, prev_macro_state)
            if record:
                producer.send(KAFKA_TOPIC, value=record)
                logger.debug(f"Sent live data for {ticker}: {record['price']}")
        
        producer.flush()
        
        # Update Previous State
        prev_macro_state = current_macro_state
        
        logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds...")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
