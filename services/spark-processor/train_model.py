
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
import logging
from datetime import timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TICKERS = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "AAPL", "GOOGL"]
# Using reliable proxy for DXY if direct DX-Y.NYB fails, but let's try standard.
# ^TNX is 10 Year Treasury
MACRO_TICKERS = ["^TNX", "DX-Y.NYB"] 
MODEL_PATH = "/app/models/fintech_model.pkl"

# Simulation Constants
TRANSACTION_COST = 0.002  # 0.2% per trade (buy + sell)
SLIPPAGE = 0.001          # 0.1% slippage
STOP_LOSS = 0.02          # 2% max loss per trade
TARGET_THRESHOLD = 0.003  # Target: Price must rise > 0.3%

def generate_synthetic_data(tickers):
    """Generates synthetic daily data (Stocks + Macro) for testing fallback."""
    logger.warning("Generarating SYNTHETIC data due to fetch failure...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=730) # 2 years
    data_frames = []
    
    # Synthetic Stocks
    for ticker in tickers:
        price = 100
        prices = []
        for _ in dates:
            change = np.random.normal(0, 1) # Normal walk
            price += change
            prices.append(max(0.1, price))
        
        df = pd.DataFrame(data={"Close": prices}, index=dates)
        df["Ticker"] = ticker
        data_frames.append(df)
        
    full_df = pd.concat(data_frames)

    # Synthetic Macro Data (join by index)
    # We cheat a bit and just add random macro columns to every row for simplicity in synthetic mode
    # Ensure index alignment
    full_df["^TNX"] = np.random.uniform(1.5, 4.5, size=len(full_df))
    full_df["DX-Y.NYB"] = np.random.uniform(90, 110, size=len(full_df))
    
    return full_df

def fetch_data(tickers):
    """Fetches Stocks and Macro data, merging them."""
    logger.info("Fetching Market Data...")
    stock_dfs = []
    
    # 1. Fetch Stocks
    try:
        for ticker in tickers:
            # Download individually to control errors
            try:
                df = yf.download(ticker, period="2y", interval="1d", progress=False)
                if not df.empty:
                    # yfinance return structure changed recently, ensuring "Close" is 1D Series
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten if MultiIndex (e.g. Price, Ticker)
                        # We try to access Close directly
                        pass
                    
                    # Basic check for structure
                    if "Close" not in df.columns and "Adj Close" in df.columns:
                         df["Close"] = df["Adj Close"]

                    # If MultiIndex with Ticker as column level, we need to extract
                    if isinstance(df.columns, pd.MultiIndex):
                         try:
                             df = df.xs(ticker, level=1, axis=1)
                         except:
                             pass # Maybe it's level 0 or not multiindex in this specific way

                    df["Ticker"] = ticker
                    # Keep only Close and Ticker
                    if "Close" in df.columns:
                        df = df[["Close", "Ticker"]] 
                        stock_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        if not stock_dfs:
            logger.error("No stock data fetched.")
            return generate_synthetic_data(tickers)
            
        full_df = pd.concat(stock_dfs)
        
        # 2. Fetch Macro Data
        logger.info("Fetching Macro Data...")
        macro_df = pd.DataFrame(index=full_df.index.unique().sort_values())
        
        for m_ticker in MACRO_TICKERS:
            try:
                m_data = yf.download(m_ticker, period="2y", interval="1d", progress=False)
                if not m_data.empty:
                    close_col = m_data["Close"] if "Close" in m_data.columns else m_data.iloc[:, 0]
                    # Handle if it's a DF
                    if isinstance(close_col, pd.DataFrame):
                        close_col = close_col.iloc[:, 0]
                        
                    clean_series = close_col.rename(m_ticker)
                    macro_df = macro_df.join(clean_series, how="left")
            except Exception as e:
                logger.warning(f"Failed to fetch macro {m_ticker}: {e}")
        
        # Fill macro gaps (forward fill last known rate)
        macro_df.ffill(inplace=True)
        macro_df.bfill(inplace=True) # Fill start if needed
        
        # Merge Macro into Stocks
        # We merge left on index. 
        # Since full_df has duplicate indices (different tickers), we join carefully.
        # join matches on index by default
        full_df = full_df.join(macro_df)
        
        return full_df

    except Exception as e:
        logger.error(f"Data Fetch Failed: {e}")
        return generate_synthetic_data(tickers)

def add_features(df):
    """Adds Technical, Macro, Time, and Sentiment features."""
    logger.info("Feature Engineering...")
    
    # Ensure sorted by Ticker and Date
    df.sort_index(inplace=True)
    
    # Initialize list to collect processed dataframes
    processed_dfs = []

    # Process each ticker group
    for ticker, group in df.groupby("Ticker"):
        group = group.copy()
        
        # 1. Log Returns
        group["Log_Ret"] = np.log(group["Close"] / group["Close"].shift(1))
        
        # 2. Lags
        lags = [1, 2, 3, 5]
        for lag in lags:
            group[f"Log_Ret_Lag{lag}"] = group["Log_Ret"].shift(lag)
            
        # 3. Technical Indicators
        # RSI
        group["RSI"] = ta.momentum.RSIIndicator(group["Close"], window=14).rsi().fillna(50)
        # MACD
        macd = ta.trend.MACD(group["Close"])
        group["MACD"] = macd.macd().fillna(0)
        group["MACD_Signal"] = macd.macd_signal().fillna(0)
        group["MACD_Diff"] = macd.macd_diff().fillna(0)
        
        # Volatility (ATR)
        atr = ta.volatility.AverageTrueRange(high=group["Close"], low=group["Close"], close=group["Close"])
        group["ATR"] = atr.average_true_range().fillna(0)
        
        # 4. Macro Interaction
        # Rate of Change of Macro vars
        if "^TNX" in group.columns:
            group["TNX_Chg"] = group["^TNX"].diff().fillna(0)
        else:
            group["TNX_Chg"] = 0
            
        if "DX-Y.NYB" in group.columns:
             group["DXY_Chg"] = group["DX-Y.NYB"].diff().fillna(0)
        else:
             group["DXY_Chg"] = 0

        # 5. Time Features
        group["DayOfWeek"] = group.index.dayofweek
        group["Month"] = group.index.month
        
        # 6. Sentiment Mock
        # In real world, merge with external sentiment DF
        group["Sentiment_Score"] = np.random.normal(0, 0.5, size=len(group))
        
        # 7. Target Generation
        # We predict if Next Day's Log Return > Threshold
        group["Log_Ret_Next"] = group["Log_Ret"].shift(-1)
        group["Target"] = (group["Log_Ret_Next"] > TARGET_THRESHOLD).astype(int)
        
        processed_dfs.append(group)
        
    df_feat = pd.concat(processed_dfs)
    
    # Drop rows with NaNs caused by lags/shifting
    df_feat.dropna(inplace=True)
    
    return df_feat

def walk_forward_validation(df):
    """
    Performs Walk-Forward Validation.
    Train on expanding window, Test on next month.
    """
    logger.info("Starting Walk-Forward Validation...")
    
    # Identify feature columns
    # Exclude non-features
    exclude_cols = ["Close", "Ticker", "Target", "Log_Ret_Next", "Log_Ret", "^TNX", "DX-Y.NYB"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Training Features: {feature_cols}")
    
    results = []
    
    # Ensure date sorted
    df.sort_index(inplace=True)
    dates = df.index.unique().sort_values()
    
    # Split into monthly chunks (approx 20 trading days)
    # Start training after 1 year (approx 250 days)
    start_idx = 250
    step_size = 20 # 1 Month
    
    total_samples = len(dates)
    
    if total_samples < start_idx + step_size:
        logger.warning("Not enough data for Walk-Forward. Training on full set only.")
        return
    
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1)
    
    total_trades = 0
    winning_trades = 0
    
    # Expanding Window Loop
    for i in range(start_idx, total_samples, step_size):
        # Define Train and Test windows
        train_end_date = dates[i]
        # Test window is from train_end to +step_size
        test_end_idx = min(i + step_size, total_samples - 1)
        test_end_date = dates[test_end_idx]
        
        if train_end_date == test_end_date:
            break

        # Masks
        train_mask = df.index <= train_end_date
        test_mask = (df.index > train_end_date) & (df.index <= test_end_date)
        
        if not test_mask.any():
            break
            
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "Target"]
        
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, "Target"]
        
        if len(X_train) < 50 or len(X_test) == 0:
            continue
            
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Evaluation Logic
        # We need to simulate result of taking the trade
        # Original DF has "Log_Ret_Next" which tells us what actually happened
        actual_returns = df.loc[test_mask, "Log_Ret_Next"].values
        
        batch_results = pd.DataFrame({
            "Target": y_test.values,
            "Pred": preds,
            "Actual_Ret": actual_returns
        })
        
        # Calculate Mock PnL
        # Rule: Buy if Pred=1. Sell next day. 
        # Cost: Entry + Exit = 0.2%
        # PnL = Return - Cost
        batch_results["Trade_PnL"] = batch_results.apply(
            lambda row: (row["Actual_Ret"] - TRANSACTION_COST) if row["Pred"] == 1 else 0, axis=1
        )
        
        results.append(batch_results)
        
        logger.info(f"Window: Train <= {train_end_date.date()} | Test <= {test_end_date.date()}")

    if not results:
        logger.warning("No results from Walk-Forward.")
        return

    full_res = pd.concat(results)
    
    # Metrics
    targets = full_res["Target"]
    preds = full_res["Pred"]
    
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    
    logger.info("="*40)
    logger.info(f"WALK-FORWARD VALIDATION SUMMARY")
    logger.info(f"Total Samples Tested: {len(full_res)}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision (Win Rate): {prec:.4f}")
    
    # Trade Analysis
    trades = full_res[full_res["Pred"] == 1]
    n_trades = len(trades)
    if n_trades > 0:
        avg_pnl = trades["Trade_PnL"].mean()
        total_pnl_percent = trades["Trade_PnL"].sum()
        win_trades = trades[trades["Trade_PnL"] > 0]
        win_rate_trades = len(win_trades) / n_trades
        
        logger.info(f"Total Trades Taken: {n_trades}")
        logger.info(f"Win Rate (After Costs): {win_rate_trades:.4f}")
        logger.info(f"Avg PnL per Trade: {avg_pnl:.5f}")
        logger.info(f"Total Simulated Return: {total_pnl_percent:.4f} (uncompounded)")
        
        # Kelly Criterion
        # f = p - q/b
        # p = Win Rate
        # b = Avg Win / Avg Loss
        if win_rate_trades > 0:
            avg_win = win_trades["Trade_PnL"].mean()
            loss_trades = trades[trades["Trade_PnL"] <= 0]
            avg_loss = abs(loss_trades["Trade_PnL"].mean()) if not loss_trades.empty else 1.0
            
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            p = win_rate_trades
            q = 1 - p
            
            kelly = p - (q / b)
            logger.info(f"Kelly Criterion (Optimal Position Size): {kelly:.2f}")
            if kelly <= 0:
                logger.warning("Kelly is Negative! Do not trade this strategy yet.")
    else:
        logger.info("No trades triggered by model.")

    logger.info("="*40)

def train_final_model():
    """Main Orchestrator."""
    # 1. Pipeline
    df = fetch_data(TICKERS)
    df = add_features(df)
    
    if df.empty:
        logger.error("Dataset empty after processing. Exiting.")
        return

    # 2. Validation
    # We run Walk-Forward to print stats for the User
    walk_forward_validation(df)
    
    # 3. Final Training (on ALL data) for Production
    logger.info("Training Final Production Model on ALL Data...")
    exclude_cols = ["Close", "Ticker", "Target", "Log_Ret_Next", "Log_Ret", "^TNX", "DX-Y.NYB"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df["Target"]
    
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    # 4. Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_final_model()
