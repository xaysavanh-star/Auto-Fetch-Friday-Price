import os
import pandas as pd
import datetime
import yfinance as yf

# 1. ດຶງຂໍ້ມູນ XAUUSD (Gold Spot)
gold = yf.Ticker("GC=F") # ຫຼື "XAUUSD=X"
data = gold.history(period="1d", interval="1m")

if not data.empty:
    latest_price = data['Close'].iloc[-1]
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_file = "friday_auto_price_log.csv"
    new_data = pd.DataFrame([[current_time, latest_price]], columns=["Timestamp", "Public_Price"])
    
    # 2. ບັນທຶກລົງ CSV
    if os.path.exists(log_file):
        new_data.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(log_file, mode='w', header=True, index=False)
    print(f"Logged: {current_time} -> {latest_price}")
else:
    print("Failed to fetch price")