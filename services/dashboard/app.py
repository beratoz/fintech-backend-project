from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine, text
import pandas as pd
import os
import json
from datetime import datetime, timedelta

# --- App Configuration ---
app = Flask(__name__)

# --- Database Connection ---
DB_URL = os.environ.get("DB_URL", "postgresql://user:password@localhost:5432/fintech")
engine = create_engine(DB_URL, pool_pre_ping=True)

# --- Period Mapping ---
PERIOD_DAYS = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "1y": 365,
    "3y": 1095
}


# --- Routes ---
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/tickers")
def get_tickers():
    """Return list of unique tickers from the database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT ticker FROM trade_signals ORDER BY ticker"))
            tickers = [row[0] for row in result]
        return jsonify(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/<ticker>")
def get_signals(ticker):
    """
    Return price/signal data for a ticker filtered by time period.
    Query param: period = 1w | 1m | 3m | 1y | 3y (default: 1m)
    """
    period = request.args.get("period", "1m")
    days = PERIOD_DAYS.get(period, 30)

    # Calculate the cutoff timestamp in milliseconds
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    try:
        query = text("""
            SELECT ticker, timestamp, price, prediction, indicators
            FROM trade_signals
            WHERE ticker = :ticker AND timestamp >= :cutoff
            ORDER BY timestamp ASC
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker, "cutoff": cutoff_ms})
            rows = result.fetchall()

        data = []
        for row in rows:
            # Parse indicators JSON
            indicators = {}
            if row[4]:
                try:
                    indicators = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                except (json.JSONDecodeError, TypeError):
                    indicators = {}

            data.append({
                "ticker": row[0],
                "timestamp": row[1],
                "price": float(row[2]) if row[2] else 0,
                "prediction": row[3],
                "indicators": indicators
            })

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/latest/<ticker>")
def get_latest(ticker):
    """Return the latest KPI data for a ticker."""
    try:
        query = text("""
            SELECT ticker, timestamp, price, prediction, indicators
            FROM trade_signals
            WHERE ticker = :ticker
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker})
            row = result.fetchone()

        if row is None:
            return jsonify({"error": "No data found"}), 404

        indicators = {}
        if row[4]:
            try:
                indicators = json.loads(row[4]) if isinstance(row[4], str) else row[4]
            except (json.JSONDecodeError, TypeError):
                indicators = {}

        return jsonify({
            "ticker": row[0],
            "timestamp": row[1],
            "price": float(row[2]) if row[2] else 0,
            "prediction": row[3],
            "indicators": indicators
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats/<ticker>")
def get_stats(ticker):
    """Return aggregate statistics for a ticker."""
    period = request.args.get("period", "1m")
    days = PERIOD_DAYS.get(period, 30)
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    try:
        query = text("""
            SELECT 
                COUNT(*) as total_signals,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as buy_signals,
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price
            FROM trade_signals
            WHERE ticker = :ticker AND timestamp >= :cutoff
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker, "cutoff": cutoff_ms})
            row = result.fetchone()

        if row is None or row[0] == 0:
            return jsonify({"error": "No data found"}), 404

        return jsonify({
            "total_signals": row[0],
            "buy_signals": row[1] or 0,
            "hold_signals": (row[0] - (row[1] or 0)),
            "min_price": float(row[2]) if row[2] else 0,
            "max_price": float(row[3]) if row[3] else 0,
            "avg_price": float(row[4]) if row[4] else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
