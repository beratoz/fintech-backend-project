import time
import json
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Constants
KAFKA_BROKER = "kafka:29092"
KAFKA_TOPIC = "stock-prices"
TICKERS = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "AAPL", "GOOGL"]
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

def fetch_history(ticker_symbol):
    """Fetches 1 year of historical data for a ticker."""
    logger.info(f"Fetching history for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1y")
        
        records = []
        for index, row in history.iterrows():
            record = {
                "ticker": ticker_symbol,
                "timestamp": int(index.timestamp() * 1000),
                "price": float(row["Close"]),
                "data_type": "HISTORY"
            }
            records.append(record)
        return records
    except Exception as e:
        logger.error(f"Error fetching history for {ticker_symbol}: {e}")
        return []

def fetch_live_price(ticker_symbol):
    """Fetches the latest live price for a ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Using 'fast_info' or fetching 1 day period to get latest
        # fast_info is often faster but 'history(period="1d")' is robust
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            latest = data.iloc[-1]
            return {
                "ticker": ticker_symbol,
                "timestamp": int(latest.name.timestamp() * 1000),
                "price": float(latest["Close"]),
                "data_type": "LIVE"
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

    # PHASE 2: Live Streaming
    while True:
        logger.info("Fetching live data cycle...")
        for ticker in TICKERS:
            record = fetch_live_price(ticker)
            if record:
                producer.send(KAFKA_TOPIC, value=record)
                logger.debug(f"Sent live data for {ticker}: {record['price']}")
        
        producer.flush()
        logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds...")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
