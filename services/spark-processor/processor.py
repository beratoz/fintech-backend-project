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
BUFFER_SIZE = 100 # Keep last 100 records for lag calculation
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
        # Sort and Truncate
        HISTORY_BUFFER[ticker].sort_index(inplace=True)
        if len(HISTORY_BUFFER[ticker]) > BUFFER_SIZE:
             HISTORY_BUFFER[ticker] = HISTORY_BUFFER[ticker].iloc[-BUFFER_SIZE:]
        
        # 3. Calculate Features (e.g. RSI, MACD) on the buffer
        # We reuse train_model.add_features BUT it recalculates macros. 
        # We need to be careful.
        # Let's effectively "Monkey Patch" or carefully copy the logic we need.
        # train_model.add_features does A LOT. Let's just run it, but ensure we have columns it needs or ignore errors.
        # BUT `add_features` expects ^TNX column to exist to calculate diff.
        # We don't have raw ^TNX. We have tnx_chg.
        # We should create a helper here or modify `add_features` to be flexible.
        # Modifying `add_features` is risky for the verify task.
        # We will manually calculate the technicals here using the same `ta` calls.
        
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
        # Incoming: tnx_chg, dxy_chg. Model expects: TNX_Chg, DXY_Chg
        full_hist["TNX_Chg"] = full_hist["tnx_chg"]
        full_hist["DXY_Chg"] = full_hist["dxy_chg"]
        
        # 5. Time
        full_hist["DayOfWeek"] = full_hist.index.dayofweek
        full_hist["Month"] = full_hist.index.month
        
        # 6. Sentiment
        full_hist["Sentiment_Score"] = full_hist["sentiment_score"]
        
        # Select Features and Predict on ONLY the new LIVE rows
        # HISTORY data is only used for buffer/technical calculations, not for predictions
        live_indices = clean_group[clean_group['data_type'] == 'LIVE'].index
        candidate_indices = full_hist.index.intersection(live_indices)
        
        if candidate_indices.empty:
            print(f"Skipping {ticker}: No LIVE data in this batch (HISTORY only, buffered for technicals)")
            continue
        
        preds_df = full_hist.loc[candidate_indices].copy()
        
        if preds_df.empty:
            continue
            
        # Feature Columns matching training
        if model:

            try:
                # DYNAMIC FEATURE ALIGNMENT
                # Ensure we use exactly the same features as training
                required_features = list(model.feature_names_in_)
                
                # Check for missing columns and fill with 0
                for feature in required_features:
                    if feature not in preds_df.columns:
                        # print(f"Warning: Missing feature {feature}, filling with 0")
                        preds_df[feature] = 0.0
                
                # Reorder columns to match model
                X = preds_df[required_features].fillna(0)
                        
                # Predict labels and probs
                predictions = model.predict(X)
                # probs = model.predict_proba(X)[:, 1] # Optional
                
                rows_to_insert = []
                
                for idx, (date, row, pred) in enumerate(zip(preds_df.index, preds_df.iterrows(), predictions)):
                    _, data = row
                    
                    # Extract indicators for JSON
                    indicators_dict = {
                        "RSI": float(data["RSI"]),
                        "MACD": float(data["MACD"]),
                        "ATR": float(data["ATR"]),
                        "Sentiment": float(data["Sentiment_Score"])
                    }
                    
                    rows_to_insert.append({
                        "ticker": str(ticker),
                        "timestamp": int(date.timestamp() * 1000), # MS timestamp
                        "price": float(data["Close"]),
                        "prediction": int(pred),
                        "indicators": json.dumps(indicators_dict)
                    })
                    
                    # Console Output
                    action = "BUY 🟢" if pred == 1 else "HOLD ⚪"
                    print(f"PREDICTION | {ticker} | {date} | Price: {data['Close']:.2f} | {action}")

                if rows_to_insert:
                    # Convert to Spark DataFrame
                    result_pdf = pd.DataFrame(rows_to_insert)
                    
                    # Define Schema for Output
                    res_schema = StructType([
                        StructField("ticker", StringType(), True),
                        StructField("timestamp", LongType(), True),
                        StructField("price", FloatType(), True),
                        StructField("prediction", IntegerType(), True),
                        StructField("indicators", StringType(), True) # JSON string
                    ])
                    
                    result_sdf = spark.createDataFrame(result_pdf, schema=res_schema)
                    
                    # Write to JDBC
                    db_url = os.environ.get("DB_URL", "jdbc:postgresql://postgres:5432/fintech")
                    if "stringtype" not in db_url:
                        db_url += "?stringtype=unspecified"

                    print(f"Writing {len(result_sdf.collect())} signals to DB...")
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
                print(f"Error processing/writing batch for {ticker}: {e}")

        else:
            print("Model not loaded, skipping prediction.")
            continue


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
