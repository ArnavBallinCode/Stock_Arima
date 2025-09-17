#!/usr/bin/env python3
"""
Demo script to test stock data loading functionality
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_data_loading():
    """Test basic data loading functionality"""
    print("🧪 Testing Stock Data Loading...")
    
    # Test stocks
    stocks = {
        "NVIDIA": "NVDA",
        "Microsoft (OpenAI proxy)": "MSFT", 
        "Meta (X proxy)": "META"
    }
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    for name, symbol in stocks.items():
        try:
            print(f"📈 Loading {name} ({symbol})...")
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if not data.empty:
                print(f"   ✅ Success! {len(data)} records loaded")
                print(f"   💰 Latest price: ${data['Close'].iloc[-1]:.2f}")
                print(f"   📊 Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
            else:
                print(f"   ⚠️  No data available")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_data_loading()
