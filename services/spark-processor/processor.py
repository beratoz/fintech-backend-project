from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
import json
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType, IntegerType
import pandas as pd
import joblib
import os
import sys

# Ensure we can import train_model
sys.path.append("/app")
import train_model

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("FintechStockProcessor") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Schema for Enriched Data
schema = StructType([
    StructField("ticker", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("price", FloatType(), True),
    StructField("data_type", StringType(), True),
    StructField("tnx_chg", FloatType(), True),
    StructField("dxy_chg", FloatType(), True),
    StructField("sentiment_score", FloatType(), True)
])

# Global State for History (In-Memory for MVP)
# { "AAPL": pd.DataFrame(...), ... }
HISTORY_BUFFER = {}
BUFFER_SIZE = 2000 # Keep enough records for good technical indicators
BUY_THRESHOLD = 0.20  # Lower than default 0.50 since model is conservative
MODEL = None
MODEL_PATH = "/app/models/fintech_model.pkl"

def load_model():
    global MODEL
    if MODEL is None:
        if os.path.exists(MODEL_PATH):
            try:
                MODEL = joblib.load(MODEL_PATH)
                print(">>> Model Loaded Successfully!")
            except Exception as e:
                print(f">>> Error Loading Model: {e}")
        else:
            print(f">>> Model not found at {MODEL_PATH}")
    return MODEL

def process_batch(batch_df, batch_id):
    """
    Process each micro-batch:
    1. Convert to Pandas
    2. Update State (History Buffer)
    3. Calculate Features
    4. Predict
    """
    print(f"--- Processing Batch {batch_id} ---")
    
    if batch_df.isEmpty():
        return

    # 1. Convert to Pandas
    pdf = batch_df.toPandas()
    
    # Ensure Timestamp is datetime for features
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], unit='ms')
    
    # Ideally Model loading should happen on Executor, but for simple foreachBatch driver/worker hybrid this works
    # If distributed, use mapPartitions or scalar iterator.
    # For now, we assume simple single-executor or driver-availabitily for MVP.
    model = load_model()

    # Process per Ticker
    for ticker, group in pdf.groupby("ticker"):
        if ticker not in HISTORY_BUFFER:
            HISTORY_BUFFER[ticker] = pd.DataFrame()
        
        # Prepare valid DF for History
        # We only really need 'Close' for Technicals + macro/sentiment for LIVE predictions.
        clean_group = group[['timestamp', 'price', 'ticker', 'data_type', 'tnx_chg', 'dxy_chg', 'sentiment_score']].copy()
        clean_group.rename(columns={'price': 'Close', 'ticker': 'Ticker'}, inplace=True)
        clean_group.set_index('timestamp', inplace=True)
        
        # Update Buffer
        HISTORY_BUFFER[ticker] = pd.concat([HISTORY_BUFFER[ticker], clean_group])
        # Deduplicate: keep only the last entry per timestamp to avoid flat features
        HISTORY_BUFFER[ticker] = HISTORY_BUFFER[ticker][~HISTORY_BUFFER[ticker].index.duplicated(keep='last')]
        # Sort and Truncate
        HISTORY_BUFFER[ticker].sort_index(inplace=True)
        if len(HISTORY_BUFFER[ticker]) > BUFFER_SIZE:
             HISTORY_BUFFER[ticker] = HISTORY_BUFFER[ticker].iloc[-BUFFER_SIZE:]
        
        # 3. Calculate Features (e.g. RSI, MACD) on the buffer
        # We manually calculate the technicals using the same `ta` calls as training.
        
        full_hist = HISTORY_BUFFER[ticker].copy()
        
        if len(full_hist) < 15: # Need at least 14 for RSI
            print(f"Skipping {ticker}: Not enough history ({len(full_hist)})")
            continue
            
        # Re-implement Feature Engineering locally to match Model input exactly
        # 1. Log Returns
        full_hist["Log_Ret"] = train_model.np.log(full_hist["Close"] / full_hist["Close"].shift(1))
        
        # 2. Lags
        lags = [1, 2, 3, 5]
        for lag in lags:
            full_hist[f"Log_Ret_Lag{lag}"] = full_hist["Log_Ret"].shift(lag)
            
        # 3. Technicals
        # RSI
        full_hist["RSI"] = train_model.ta.momentum.RSIIndicator(full_hist["Close"], window=14).rsi().fillna(50)
        # MACD
        macd = train_model.ta.trend.MACD(full_hist["Close"])
        full_hist["MACD"] = macd.macd().fillna(0)
        full_hist["MACD_Signal"] = macd.macd_signal().fillna(0)
        full_hist["MACD_Diff"] = macd.macd_diff().fillna(0)
        
        # ATR
        atr = train_model.ta.volatility.AverageTrueRange(high=full_hist["Close"], low=full_hist["Close"], close=full_hist["Close"])
        full_hist["ATR"] = atr.average_true_range().fillna(0)
        
        # 4. Macro (Already have Chg, map to correct names)
        full_hist["TNX_Chg"] = full_hist["tnx_chg"]
        full_hist["DXY_Chg"] = full_hist["dxy_chg"]
        
        # 5. Time
        full_hist["DayOfWeek"] = full_hist.index.dayofweek
        full_hist["Month"] = full_hist.index.month
        
        # 6. Sentiment
        full_hist["Sentiment_Score"] = full_hist["sentiment_score"]
        
        # ── Write ALL new rows to DB with ML predictions ────────────────
        # Get only rows that are new in this batch
        new_indices = clean_group.index
        batch_indices = full_hist.index.intersection(new_indices)
        batch_df_to_write = full_hist.loc[batch_indices].copy()
        
        # Deduplicate batch by index (timestamp) — keep last
        batch_df_to_write = batch_df_to_write[~batch_df_to_write.index.duplicated(keep='last')]
        
        if batch_df_to_write.empty:
            continue
        
        # Default prediction = 0 (HOLD)
        batch_df_to_write['_prediction'] = 0
        
        # Run ML model on ALL rows that have enough feature history
        if model:
            try:
                required_features = list(model.feature_names_in_)
                
                # Ensure all required features exist
                for feature in required_features:
                    if feature not in batch_df_to_write.columns:
                        batch_df_to_write[feature] = 0.0
                
                # Use predict_proba with lower threshold for BUY signals
                predict_df = batch_df_to_write[required_features].fillna(0)
                probas = model.predict_proba(predict_df)[:, 1]  # probability of class 1 (BUY)
                predictions = (probas >= BUY_THRESHOLD).astype(int)
                batch_df_to_write['_prediction'] = predictions
                
                buy_count = sum(predictions == 1)
                hold_count = sum(predictions == 0)
                print(f"ML | {ticker} | {len(predictions)} predictions: {buy_count} BUY, {hold_count} HOLD (max_proba={probas.max():.3f})")
                    
            except Exception as e:
                print(f"Error predicting for {ticker}: {e}")
        
        # Build rows for DB insertion
        rows_to_insert = []
        for date, data in batch_df_to_write.iterrows():
            indicators_dict = {
                "RSI": float(data.get("RSI", 50)),
                "MACD": float(data.get("MACD", 0)),
                "ATR": float(data.get("ATR", 0)),
                "Sentiment": float(data.get("Sentiment_Score", 0))
            }
            rows_to_insert.append({
                "ticker": str(ticker),
                "timestamp": int(date.timestamp() * 1000),
                "price": float(data["Close"]),
                "prediction": int(data['_prediction']),
                "indicators": json.dumps(indicators_dict)
            })
        
        if rows_to_insert:
            try:
                result_pdf = pd.DataFrame(rows_to_insert)
                
                res_schema = StructType([
                    StructField("ticker", StringType(), True),
                    StructField("timestamp", LongType(), True),
                    StructField("price", FloatType(), True),
                    StructField("prediction", IntegerType(), True),
                    StructField("indicators", StringType(), True)
                ])
                
                result_sdf = spark.createDataFrame(result_pdf, schema=res_schema)
                
                db_url = os.environ.get("DB_URL", "jdbc:postgresql://postgres:5432/fintech")
                if "stringtype" not in db_url:
                    db_url += "?stringtype=unspecified"

                print(f"Writing {len(rows_to_insert)} unique records for {ticker} to DB...")
                result_sdf.write \
                    .format("jdbc") \
                    .option("url", db_url) \
                    .option("dbtable", "trade_signals") \
                    .option("user", os.environ.get("DB_USER", "user")) \
                    .option("password", os.environ.get("DB_PASSWORD", "password")) \
                    .option("driver", "org.postgresql.Driver") \
                    .mode("append") \
                    .save()
                    
            except Exception as e:
                print(f"Error writing batch for {ticker}: {e}")


# Read Stream
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "stock-prices") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Write Stream using foreachBatch
query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()
