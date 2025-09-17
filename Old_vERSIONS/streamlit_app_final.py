import os
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# For modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# TensorFlow import with proper configuration
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU only and disable GPU
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    st.warning(f"TensorFlow not available: {e}. LSTM predictions will be disabled.")

# Statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Complete Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .model-metrics {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Complete Stock Analysis & ML Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar for stock selection and parameters
st.sidebar.header("🔧 Configuration")

# Stock selection
stock_options = {
    "NVIDIA": "NVDA",
    "OpenAI (Microsoft)": "MSFT",
    "X (Twitter) - Meta": "META"
}

selected_stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
stock_symbol = stock_options[selected_stock_name]

# Date range selection
st.sidebar.subheader("📅 Date Range")
end_date = st.sidebar.date_input("End Date", datetime.now())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

# Analysis period selection
st.sidebar.subheader("🔍 Analysis Period")
analysis_type = st.sidebar.radio("Select Analysis Period Type", ["Specific Day", "Specific Week", "Custom Range"])

if analysis_type == "Specific Day":
    analysis_date = st.sidebar.date_input("Select Day for Analysis", end_date)
elif analysis_type == "Specific Week":
    week_start = st.sidebar.date_input("Week Start Date", end_date - timedelta(days=7))
    analysis_date = week_start
else:
    analysis_start = st.sidebar.date_input("Analysis Start", end_date - timedelta(days=30))
    analysis_end = st.sidebar.date_input("Analysis End", end_date)

# Load data function
@st.cache_data
def load_stock_data(symbol, start, end):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data preprocessing function
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Original data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Original Data Info")
        st.write(f"Shape: {df.shape}")
        st.write(f"Date Range: {df.index.min()} to {df.index.max()}")
        
    with col2:
        st.subheader("🔢 Data Types")
        st.write(df.dtypes)
    
    # Check for missing values
    st.subheader("🔍 Missing Values Analysis")
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        st.warning("Missing values found!")
        st.write(missing_values)
        
        # Fill missing values
        fill_method = st.selectbox("Select fill method", ["Forward Fill", "Backward Fill", "Interpolate", "Mean"])
        
        if fill_method == "Forward Fill":
            df = df.fillna(method='ffill')
        elif fill_method == "Backward Fill":
            df = df.fillna(method='bfill')
        elif fill_method == "Interpolate":
            df = df.interpolate()
        else:
            df = df.fillna(df.mean())
            
        st.success("✅ Missing values filled!")
    else:
        st.success("✅ No missing values found!")
    
    # Remove duplicates and other preprocessing steps
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_shape - df.shape[0]
    
    if removed_duplicates > 0:
        st.info(f"🔄 Removed {removed_duplicates} duplicate rows")
    else:
        st.success("✅ No duplicate rows found!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return df

# Enhanced analysis function (simplified for space)
def enhanced_analysis(df, analysis_type, analysis_date=None, analysis_start=None, analysis_end=None):
    """Enhanced analysis with multiple chart types"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Enhanced Data Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data based on analysis period
    if analysis_type == "Specific Day":
        analysis_df = df[df.index.date == analysis_date]
        period_name = f"Day: {analysis_date}"
    elif analysis_type == "Specific Week":
        week_end = analysis_date + timedelta(days=6)
        analysis_df = df[(df.index.date >= analysis_date) & (df.index.date <= week_end)]
        period_name = f"Week: {analysis_date} to {week_end}"
    else:
        analysis_df = df[(df.index.date >= analysis_start) & (df.index.date <= analysis_end)]
        period_name = f"Period: {analysis_start} to {analysis_end}"
    
    if analysis_df.empty:
        st.warning("No data available for the selected period!")
        return
    
    st.subheader(f"📈 Analysis for {period_name}")
    
    # Calculate comprehensive metrics
    analysis_df['Daily_Return'] = analysis_df['Close'].pct_change()
    analysis_df['Price_Range'] = analysis_df['High'] - analysis_df['Low']
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=min(5, len(analysis_df))).mean()
    
    # Key metrics
    price_change = analysis_df['Close'].iloc[-1] - analysis_df['Close'].iloc[0]
    price_change_pct = (price_change / analysis_df['Close'].iloc[0]) * 100
    volatility = analysis_df['Close'].std()
    volume_avg = analysis_df['Volume'].mean()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("📊 Volatility", f"${volatility:.2f}")
    with col3:
        st.metric("📦 Avg Volume", f"{volume_avg:,.0f}")
    with col4:
        avg_return = analysis_df['Daily_Return'].mean() * 100 if len(analysis_df) > 1 else 0
        st.metric("📈 Avg Daily Return", f"{avg_return:.3f}%")
    
    # Candlestick chart
    fig_candle = go.Figure()
    
    fig_candle.add_trace(go.Candlestick(
        x=analysis_df.index,
        open=analysis_df['Open'],
        high=analysis_df['High'],
        low=analysis_df['Low'],
        close=analysis_df['Close'],
        name="Price"
    ))
    
    fig_candle.update_layout(
        title="📈 Candlestick Chart",
        yaxis_title="Price ($)",
        height=500
    )
    st.plotly_chart(fig_candle, use_container_width=True)
    
    # Volume chart
    if len(analysis_df) > 1:
        fig_vol = px.bar(
            analysis_df,
            x=analysis_df.index,
            y='Volume',
            title="📦 Trading Volume",
            color='Volume',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return analysis_df

# LSTM Model with proper TensorFlow handling
def build_lstm_model(data, look_back=60):
    """Build and train LSTM model with accuracy metrics"""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available. Cannot build LSTM model.")
        return None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🧠 LSTM Neural Network Model</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Create sequences
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, look_back)
        
        if len(X) < 20:
            st.warning("⚠️ Not enough data for LSTM training. Need at least 80 data points.")
            st.markdown('</div>', unsafe_allow_html=True)
            return None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with explicit CPU device
        with tf.device('/CPU:0'):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
        
        # Compile with explicit device
        with tf.device('/CPU:0'):
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        with st.spinner("🎯 Training LSTM model..."):
            with tf.device('/CPU:0'):
                history = model.fit(
                    X_train, y_train, 
                    batch_size=16, 
                    epochs=30, 
                    verbose=0, 
                    validation_split=0.1
                )
        
        # Make predictions
        with tf.device('/CPU:0'):
            train_predict = model.predict(X_train, verbose=0)
            test_predict = model.predict(X_test, verbose=0)
        
        # Transform back to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate comprehensive metrics
        train_mse = mean_squared_error(y_train_actual, train_predict)
        test_mse = mean_squared_error(y_test_actual, test_predict)
        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        train_r2 = r2_score(y_train_actual, train_predict)
        test_r2 = r2_score(y_test_actual, test_predict)
        
        # Calculate accuracy percentage (1 - normalized error)
        train_accuracy = max(0, (1 - (train_mae / y_train_actual.mean())) * 100)
        test_accuracy = max(0, (1 - (test_mae / y_test_actual.mean())) * 100)
        
        # Display metrics in a structured way
        st.markdown('<div class="model-metrics">', unsafe_allow_html=True)
        st.subheader("🎯 LSTM Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Accuracy Metrics**")
            st.metric("🎯 Training Accuracy", f"{train_accuracy:.2f}%")
            st.metric("🎯 Test Accuracy", f"{test_accuracy:.2f}%")
        
        with col2:
            st.write("**📈 Error Metrics**")
            st.metric("📊 Train MAE", f"${train_mae:.2f}")
            st.metric("📊 Test MAE", f"${test_mae:.2f}")
        
        with col3:
            st.write("**🔍 Statistical Metrics**")
            st.metric("📉 Train R²", f"{train_r2:.3f}")
            st.metric("📉 Test R²", f"{test_r2:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot predictions
        fig = go.Figure()
        
        # Actual prices
        train_dates = data.index[look_back:look_back+len(y_train_actual)]
        test_dates = data.index[look_back+len(y_train_actual):look_back+len(y_train_actual)+len(y_test_actual)]
        
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=y_train_actual.flatten(),
            mode='lines',
            name='Actual (Train)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual (Test)',
            line=dict(color='green', width=2)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_predict.flatten(),
            mode='lines',
            name=f'LSTM Train Pred (Acc: {train_accuracy:.1f}%)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict.flatten(),
            mode='lines',
            name=f'LSTM Test Pred (Acc: {test_accuracy:.1f}%)',
            line=dict(color='orange', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="🧠 LSTM Model: Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 LSTM Future Prediction (Next Day)")
        
        # Get last sequence for prediction
        last_sequence = scaled_data[-look_back:]
        with tf.device('/CPU:0'):
            future_pred_scaled = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        future_pred = scaler.inverse_transform(future_pred_scaled)[0, 0]
        
        change = future_pred - data['Close'].iloc[-1]
        change_pct = (change / data['Close'].iloc[-1]) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.write(f"{direction} **LSTM Prediction**: **${future_pred:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        st.write(f"🎯 **Model Accuracy**: {test_accuracy:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model, scaler, test_accuracy
        
    except Exception as e:
        st.error(f"❌ Error in LSTM model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None, 0

# Enhanced ML models with accuracy metrics
def build_ml_models(data):
    """Build ML models with comprehensive accuracy metrics"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        df = data.copy()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        if len(df) < 10:
            st.warning("⚠️ Not enough data for ML predictions.")
            return None
        
        # Features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features].values
        y = df['Target'].values
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predict
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate comprehensive metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = max(0, (1 - (mae / y_test.mean())) * 100)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'accuracy': accuracy
            }
        
        # Display results
        st.markdown('<div class="model-metrics">', unsafe_allow_html=True)
        st.subheader("🎯 ML Models Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        for i, (name, result) in enumerate(results.items()):
            with col1 if i == 0 else col2:
                st.write(f"**🤖 {name}**")
                st.metric("🎯 Accuracy", f"{result['accuracy']:.2f}%")
                st.metric("📊 MAE", f"${result['mae']:.2f}")
                st.metric("📉 R² Score", f"{result['r2']:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot predictions
        fig = go.Figure()
        
        test_dates = df.index[train_size:]
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test,
            mode='lines',
            name='Actual Prices',
            line=dict(color='blue', width=3)
        ))
        
        # Predictions
        colors = ['red', 'orange']
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=result['predictions'],
                mode='lines',
                name=f'{name} (Acc: {result["accuracy"]:.1f}%)',
                line=dict(color=colors[i], dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="🤖 ML Models: Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 ML Models Future Predictions (Next Day)")
        
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler_X.transform(last_features)
        
        for name, result in results.items():
            future_pred_scaled = result['model'].predict(last_features_scaled)
            future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1))[0, 0]
            
            change = future_pred - data['Close'].iloc[-1]
            change_pct = (change / data['Close'].iloc[-1]) * 100
            direction = "📈" if change > 0 else "📉"
            
            st.write(f"{direction} **{name}**: **${future_pred:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%) - Accuracy: {result['accuracy']:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return results
        
    except Exception as e:
        st.error(f"❌ Error in ML models: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None

# ARIMA Model with accuracy metrics
def build_arima_model(data):
    """Build ARIMA model with accuracy metrics"""
    if not STATSMODELS_AVAILABLE:
        st.error("❌ Statsmodels not available. Cannot build ARIMA model.")
        return None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 ARIMA Statistical Model</h2>', unsafe_allow_html=True)
    
    try:
        prices = data['Close']
        
        # Split data
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Fit ARIMA model
        with st.spinner("🎯 Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=(1, 1, 1))
            fitted_arima = arima_model.fit()
        
        # Make predictions
        forecast = fitted_arima.forecast(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecast)
        mae = mean_absolute_error(test_data, forecast)
        r2 = r2_score(test_data, forecast)
        accuracy = max(0, (1 - (mae / test_data.mean())) * 100)
        
        # Display metrics
        st.markdown('<div class="model-metrics">', unsafe_allow_html=True)
        st.subheader("🎯 ARIMA Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.2f}%")
        with col2:
            st.metric("📊 MAE", f"${mae:.2f}")
        with col3:
            st.metric("📉 R² Score", f"{r2:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot results
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_data.index, y=train_data.values,
            mode='lines', name='Training Data', line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values,
            mode='lines', name='Actual Test Data', line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index, y=forecast,
            mode='lines', name=f'ARIMA Forecast (Acc: {accuracy:.1f}%)', 
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="📊 ARIMA Model: Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 ARIMA Future Prediction (Next Day)")
        future_forecast = fitted_arima.forecast(steps=1)[0]
        change = future_forecast - data['Close'].iloc[-1]
        change_pct = (change / data['Close'].iloc[-1]) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.write(f"{direction} **ARIMA Prediction**: **${future_forecast:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%) - Accuracy: {accuracy:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return fitted_arima, accuracy
        
    except Exception as e:
        st.error(f"❌ Error fitting ARIMA model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, 0

# Main application
def main():
    # Load data
    if st.sidebar.button("🔄 Load Data", use_container_width=True):
        with st.spinner("📥 Loading stock data..."):
            data = load_stock_data(stock_symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            st.session_state['data'] = data
            st.session_state['stock_name'] = selected_stock_name
            st.success(f"✅ Data loaded successfully for **{selected_stock_name}** ({stock_symbol})")
            
            # Show basic data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Records", len(data))
            with col2:
                st.metric("📅 Date Range", f"{(data.index[-1] - data.index[0]).days} days")
            with col3:
                st.metric("💰 Latest Price", f"${data['Close'].iloc[-1]:.2f}")
        else:
            st.error("❌ Failed to load data. Please check your internet connection and try again.")
    
    # Process data if loaded
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Display raw data
        st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
        st.dataframe(data.tail(10), use_container_width=True)
        
        # Data preprocessing
        if st.button("🔧 Preprocess Data", use_container_width=True):
            processed_data = preprocess_data(data.copy())
            st.session_state['processed_data'] = processed_data
        
        # Analysis
        if 'processed_data' in st.session_state:
            processed_data = st.session_state['processed_data']
            
            if st.button("📊 Run Analysis", use_container_width=True):
                if analysis_type == "Specific Day":
                    analysis_df = enhanced_analysis(processed_data, analysis_type, analysis_date)
                elif analysis_type == "Specific Week":
                    analysis_df = enhanced_analysis(processed_data, analysis_type, analysis_date)
                else:
                    analysis_df = enhanced_analysis(processed_data, analysis_type, 
                                                 analysis_start=analysis_start, analysis_end=analysis_end)
                
                if analysis_df is not None:
                    st.session_state['analysis_df'] = analysis_df
            
            # Model prediction section
            st.markdown('<h2 class="section-header">🔮 ML Prediction Models with Accuracy</h2>', unsafe_allow_html=True)
            
            model_choice = st.selectbox("🤖 Select Models to Run", 
                                       ["All Models", "LSTM Only", "Traditional ML Only", "ARIMA Only"])
            
            if st.button("🚀 Train Models & Predict", use_container_width=True):
                if model_choice in ["All Models", "Traditional ML Only"]:
                    ml_results = build_ml_models(processed_data)
                
                if model_choice in ["All Models", "LSTM Only"]:
                    lstm_model, scaler, lstm_accuracy = build_lstm_model(processed_data)
                
                if model_choice in ["All Models", "ARIMA Only"]:
                    arima_model, arima_accuracy = build_arima_model(processed_data)
                
                # Model comparison summary
                if model_choice == "All Models":
                    st.markdown('<h3 class="section-header">🏆 Model Accuracy Comparison</h3>', unsafe_allow_html=True)
                    
                    comparison_data = []
                    if 'ml_results' in locals() and ml_results:
                        for name, result in ml_results.items():
                            comparison_data.append({
                                'Model': name,
                                'Accuracy': f"{result['accuracy']:.2f}%",
                                'MAE': f"${result['mae']:.2f}",
                                'Type': 'Traditional ML'
                            })
                    
                    if 'lstm_accuracy' in locals():
                        comparison_data.append({
                            'Model': 'LSTM Neural Network',
                            'Accuracy': f"{lstm_accuracy:.2f}%",
                            'MAE': 'See above',
                            'Type': 'Deep Learning'
                        })
                    
                    if 'arima_accuracy' in locals():
                        comparison_data.append({
                            'Model': 'ARIMA Statistical',
                            'Accuracy': f"{arima_accuracy:.2f}%",
                            'MAE': 'See above',
                            'Type': 'Statistical'
                        })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
