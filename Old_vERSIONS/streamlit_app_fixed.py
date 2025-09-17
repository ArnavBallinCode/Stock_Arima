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
import warnings
warnings.filterwarnings('ignore')

# For modeling - with safe imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow import with error handling
try:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')  # Force CPU usage
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. LSTM predictions will be disabled.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("Statsmodels not available. ARIMA predictions will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Analysis Platform",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Enhanced Stock Analysis & Prediction Platform</h1>', unsafe_allow_html=True)

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
    
    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_shape - df.shape[0]
    
    if removed_duplicates > 0:
        st.info(f"🔄 Removed {removed_duplicates} duplicate rows")
    else:
        st.success("✅ No duplicate rows found!")
    
    # Data validation
    st.subheader("✅ Data Validation")
    
    # Check for negative prices
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if negative_prices:
        st.warning("⚠️ Warning: Negative prices found in data!")
    else:
        st.success("✅ All prices are positive")
    
    # Check for logical price relationships
    price_logic = ((df['High'] >= df['Low']) & 
                   (df['High'] >= df['Open']) & 
                   (df['High'] >= df['Close']) &
                   (df['Low'] <= df['Open']) & 
                   (df['Low'] <= df['Close'])).all()
    
    if price_logic:
        st.success("✅ Price relationships are logical (High >= Open, Close, Low)")
    else:
        st.warning("⚠️ Warning: Some price relationships seem illogical!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return df

# Enhanced analysis function with comprehensive visualizations
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
    analysis_df['Body_Size'] = abs(analysis_df['Close'] - analysis_df['Open'])
    analysis_df['Upper_Shadow'] = analysis_df['High'] - analysis_df[['Open', 'Close']].max(axis=1)
    analysis_df['Lower_Shadow'] = analysis_df[['Open', 'Close']].min(axis=1) - analysis_df['Low']
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=5).mean()
    analysis_df['Price_MA'] = analysis_df['Close'].rolling(window=5).mean()
    
    # Key metrics
    price_change = analysis_df['Close'].iloc[-1] - analysis_df['Close'].iloc[0]
    price_change_pct = (price_change / analysis_df['Close'].iloc[0]) * 100
    volatility = analysis_df['Close'].std()
    volume_avg = analysis_df['Volume'].mean()
    max_price = analysis_df['High'].max()
    min_price = analysis_df['Low'].min()
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("📊 Volatility", f"${volatility:.2f}")
    with col3:
        st.metric("📦 Avg Volume", f"{volume_avg:,.0f}")
    with col4:
        st.metric("📏 Price Range", f"${min_price:.2f} - ${max_price:.2f}")
    with col5:
        avg_return = analysis_df['Daily_Return'].mean() * 100
        st.metric("📈 Avg Daily Return", f"{avg_return:.3f}%")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Analysis", "📊 Volume Analysis", "🎯 Technical Analysis", "📉 Risk Analysis", "🔍 Pattern Analysis"])
    
    with tab1:
        st.subheader("📈 Price Movement Analysis")
        
        # Price charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns bar chart
            fig_returns = px.bar(
                analysis_df.dropna(),
                x=analysis_df.dropna().index,
                y='Daily_Return',
                title="Daily Returns",
                color='Daily_Return',
                color_continuous_scale='RdYlGn'
            )
            fig_returns.update_layout(height=400)
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Price range analysis
            fig_range = px.bar(
                analysis_df,
                x=analysis_df.index,
                y='Price_Range',
                title="Daily Price Range (High - Low)",
                color='Price_Range',
                color_continuous_scale='Viridis'
            )
            fig_range.update_layout(height=400)
            st.plotly_chart(fig_range, use_container_width=True)
        
        # Candlestick with moving averages
        fig_candle = go.Figure()
        
        fig_candle.add_trace(go.Candlestick(
            x=analysis_df.index,
            open=analysis_df['Open'],
            high=analysis_df['High'],
            low=analysis_df['Low'],
            close=analysis_df['Close'],
            name="Price"
        ))
        
        if len(analysis_df) > 5:
            fig_candle.add_trace(go.Scatter(
                x=analysis_df.index,
                y=analysis_df['Price_MA'],
                mode='lines',
                name='5-Day MA',
                line=dict(color='orange', width=2)
            ))
        
        fig_candle.update_layout(
            title="Candlestick Chart with Moving Average",
            yaxis_title="Price ($)",
            height=500
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    
    with tab2:
        st.subheader("📦 Volume Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume bar chart
            fig_vol = px.bar(
                analysis_df,
                x=analysis_df.index,
                y='Volume',
                title="Trading Volume",
                color='Volume',
                color_continuous_scale='Blues'
            )
            fig_vol.update_layout(height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Volume moving average
            if len(analysis_df) > 5:
                fig_vol_ma = go.Figure()
                fig_vol_ma.add_trace(go.Scatter(
                    x=analysis_df.index,
                    y=analysis_df['Volume'],
                    mode='lines',
                    name='Volume',
                    line=dict(color='blue')
                ))
                fig_vol_ma.add_trace(go.Scatter(
                    x=analysis_df.index,
                    y=analysis_df['Volume_MA'],
                    mode='lines',
                    name='Volume 5-Day MA',
                    line=dict(color='red', width=2)
                ))
                fig_vol_ma.update_layout(
                    title="Volume vs Moving Average",
                    yaxis_title="Volume",
                    height=400
                )
                st.plotly_chart(fig_vol_ma, use_container_width=True)
        
        # Price vs Volume correlation
        fig_corr = px.scatter(
            analysis_df,
            x='Volume',
            y='Close',
            size='Price_Range',
            title="Price vs Volume Correlation",
            labels={'Volume': 'Trading Volume', 'Close': 'Closing Price ($)'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Technical Analysis")
        
        # Technical indicators
        if len(analysis_df) > 20:
            # Bollinger Bands
            analysis_df['BB_Middle'] = analysis_df['Close'].rolling(window=20).mean()
            analysis_df['BB_Std'] = analysis_df['Close'].rolling(window=20).std()
            analysis_df['BB_Upper'] = analysis_df['BB_Middle'] + (analysis_df['BB_Std'] * 2)
            analysis_df['BB_Lower'] = analysis_df['BB_Middle'] - (analysis_df['BB_Std'] * 2)
            
            fig_bb = go.Figure()
            
            fig_bb.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['Close'],
                mode='lines', name='Close Price', line=dict(color='blue')
            ))
            fig_bb.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['BB_Upper'],
                mode='lines', name='Upper Band', line=dict(color='red', dash='dash')
            ))
            fig_bb.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['BB_Lower'],
                mode='lines', name='Lower Band', line=dict(color='red', dash='dash'),
                fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
            ))
            fig_bb.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['BB_Middle'],
                mode='lines', name='Middle Band (20-MA)', line=dict(color='orange')
            ))
            
            fig_bb.update_layout(
                title="Bollinger Bands",
                yaxis_title="Price ($)",
                height=500
            )
            st.plotly_chart(fig_bb, use_container_width=True)
        
        # RSI calculation and display
        if len(analysis_df) > 14:
            delta = analysis_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            analysis_df['RSI'] = 100 - (100 / (1 + rs))
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['RSI'],
                mode='lines', name='RSI', line=dict(color='purple')
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(
                title="Relative Strength Index (RSI)",
                yaxis_title="RSI",
                height=400
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    with tab4:
        st.subheader("📉 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            if 'Daily_Return' in analysis_df.columns:
                fig_dist = px.histogram(
                    analysis_df.dropna(),
                    x='Daily_Return',
                    nbins=20,
                    title="Daily Returns Distribution",
                    labels={'Daily_Return': 'Daily Return', 'count': 'Frequency'}
                )
                fig_dist.add_vline(x=analysis_df['Daily_Return'].mean(), 
                                 line_dash="dash", line_color="red", 
                                 annotation_text="Mean")
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Risk metrics
            returns = analysis_df['Daily_Return'].dropna()
            if len(returns) > 0:
                risk_metrics = {
                    'Volatility (Annual)': f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                    'Sharpe Ratio': f"{returns.mean() / returns.std() * np.sqrt(252):.2f}" if returns.std() > 0 else "N/A",
                    'Max Daily Gain': f"{returns.max() * 100:.2f}%",
                    'Max Daily Loss': f"{returns.min() * 100:.2f}%",
                    'Positive Days': f"{(returns > 0).sum()} / {len(returns)}",
                    'VaR (95%)': f"{np.percentile(returns, 5) * 100:.2f}%"
                }
                
                st.subheader("📊 Risk Metrics")
                for metric, value in risk_metrics.items():
                    st.write(f"**{metric}**: {value}")
    
    with tab5:
        st.subheader("🔍 Pattern Analysis")
        
        # Candlestick patterns analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Body vs Shadow analysis
            fig_body = px.scatter(
                analysis_df,
                x='Body_Size',
                y='Upper_Shadow',
                size='Volume',
                color='Daily_Return',
                title="Candlestick Body vs Upper Shadow",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_body, use_container_width=True)
        
        with col2:
            # Pattern identification
            patterns = {
                'Doji': (analysis_df['Body_Size'] < (analysis_df['Price_Range'] * 0.1)).sum(),
                'Long Body': (analysis_df['Body_Size'] > (analysis_df['Price_Range'] * 0.7)).sum(),
                'Upper Shadow Dominant': (analysis_df['Upper_Shadow'] > analysis_df['Lower_Shadow']).sum(),
                'Lower Shadow Dominant': (analysis_df['Lower_Shadow'] > analysis_df['Upper_Shadow']).sum(),
            }
            
            fig_patterns = px.pie(
                values=list(patterns.values()),
                names=list(patterns.keys()),
                title="Candlestick Patterns Distribution"
            )
            st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Volume-Price relationship
        if len(analysis_df) > 1:
            price_volume_corr = analysis_df['Close'].corr(analysis_df['Volume'])
            st.write(f"**Price-Volume Correlation**: {price_volume_corr:.3f}")
            
            if abs(price_volume_corr) > 0.5:
                st.success("Strong correlation between price and volume")
            elif abs(price_volume_corr) > 0.3:
                st.info("Moderate correlation between price and volume")
            else:
                st.warning("Weak correlation between price and volume")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return analysis_df

# LSTM Model with proper error handling
def build_lstm_model(data, look_back=60):
    """Build and train LSTM model with proper error handling"""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available. Cannot build LSTM model.")
        return None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 LSTM Neural Network Prediction</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Create training data
        def create_sequences(data, look_back):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, look_back)
        
        if len(X) < 10:
            st.warning("⚠️ Not enough data for LSTM training. Need at least 70 data points.")
            return None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with CPU enforcement
        with tf.device('/CPU:0'):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        with st.spinner("🎯 Training LSTM model... This may take a moment."):
            history = model.fit(X_train, y_train, batch_size=16, epochs=25, verbose=0, validation_split=0.1)
        
        # Make predictions
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        
        # Transform back to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_actual, train_predict)
        test_mse = mean_squared_error(y_test_actual, test_predict)
        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Train MSE", f"{train_mse:.2f}")
        with col2:
            st.metric("🎯 Test MSE", f"{test_mse:.2f}")
        with col3:
            st.metric("📊 Train MAE", f"${train_mae:.2f}")
        with col4:
            st.metric("📊 Test MAE", f"${test_mae:.2f}")
        
        # Plot predictions
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=data.index[look_back:look_back+len(y_train_actual)],
            y=y_train_actual.flatten(),
            mode='lines',
            name='Actual (Train)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index[look_back+len(y_train_actual):look_back+len(y_train_actual)+len(y_test_actual)],
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual (Test)',
            line=dict(color='green', width=2)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=data.index[look_back:look_back+len(train_predict)],
            y=train_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Train)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)],
            y=test_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Test)',
            line=dict(color='orange', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="🤖 LSTM Model Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Predictions (Next 5 days)")
        
        # Get last sequence
        last_sequence = scaled_data[-look_back:]
        future_predictions = []
        
        for i in range(5):
            next_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
        
        # Display future predictions
        st.write("📅 **LSTM Predictions:**")
        for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
            change = price[0] - data['Close'].iloc[-1]
            change_pct = (change / data['Close'].iloc[-1]) * 100
            direction = "📈" if change > 0 else "📉"
            st.write(f"{direction} **Day {i+1}** ({date.strftime('%Y-%m-%d')}): **${price[0]:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error in LSTM model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None

# ARIMA Model with enhanced features
def build_arima_model(data):
    """Build and train ARIMA model with enhanced features"""
    if not STATSMODELS_AVAILABLE:
        st.error("❌ Statsmodels not available. Cannot build ARIMA model.")
        return None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 ARIMA Statistical Model Prediction</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        prices = data['Close']
        
        # Check stationarity
        st.subheader("📈 Stationarity Analysis")
        result = adfuller(prices.dropna())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ADF Statistic**: {result[0]:.6f}")
            st.write(f"**p-value**: {result[1]:.6f}")
        
        with col2:
            if result[1] <= 0.05:
                st.success("✅ Series is stationary")
                diff_order = 0
            else:
                st.warning("⚠️ Series is not stationary. Applying differencing...")
                diff_order = 1
        
        # If not stationary, difference the series
        if diff_order > 0:
            prices_diff = prices.diff().dropna()
            result_diff = adfuller(prices_diff)
            st.write(f"**After differencing** - ADF Statistic: {result_diff[0]:.6f}, p-value: {result_diff[1]:.6f}")
        
        # Split data
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Auto ARIMA parameter selection
        st.subheader("🔧 ARIMA Parameter Selection")
        auto_arima = st.checkbox("Use Auto ARIMA (recommended)", value=True)
        
        if auto_arima:
            # Simple parameter search
            best_aic = float('inf')
            best_params = None
            
            with st.spinner("🔍 Finding optimal ARIMA parameters..."):
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                model = ARIMA(train_data, order=(p, d, q))
                                fitted_model = model.fit()
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_params = (p, d, q)
                            except:
                                continue
            
            if best_params:
                st.success(f"✅ **Best ARIMA parameters**: {best_params} (AIC: {best_aic:.2f})")
            else:
                best_params = (1, 1, 1)
                st.warning("⚠️ Could not find optimal parameters. Using default (1,1,1)")
        else:
            p = st.slider("AR order (p)", 0, 5, 1)
            d = st.slider("Differencing order (d)", 0, 2, 1)
            q = st.slider("MA order (q)", 0, 5, 1)
            best_params = (p, d, q)
        
        # Fit ARIMA model
        with st.spinner("🎯 Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=best_params)
            fitted_arima = arima_model.fit()
        
        # Model summary in expandable section
        with st.expander("📋 View Model Summary"):
            st.text(str(fitted_arima.summary()))
        
        # Make predictions
        forecast_steps = len(test_data)
        forecast = fitted_arima.forecast(steps=forecast_steps)
        forecast_ci = fitted_arima.get_forecast(steps=forecast_steps).conf_int()
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecast)
        mae = mean_absolute_error(test_data, forecast)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Test MSE", f"{mse:.2f}")
        with col2:
            st.metric("📊 Test MAE", f"${mae:.2f}")
        
        # Plot results
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data.values,
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Actual Test Data',
            line=dict(color='green', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=forecast,
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=forecast_ci.iloc[:, 0],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=forecast_ci.iloc[:, 1],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='95% Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title="📊 ARIMA Model Predictions with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Predictions (Next 5 days)")
        future_forecast = fitted_arima.forecast(steps=5)
        future_ci = fitted_arima.get_forecast(steps=5).conf_int()
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
        
        st.write("📅 **ARIMA Predictions:**")
        for i, (date, price, ci_lower, ci_upper) in enumerate(zip(future_dates, future_forecast, 
                                                                  future_ci.iloc[:, 0], future_ci.iloc[:, 1])):
            change = price - data['Close'].iloc[-1]
            change_pct = (change / data['Close'].iloc[-1]) * 100
            direction = "📈" if change > 0 else "📉"
            st.write(f"{direction} **Day {i+1}** ({date.strftime('%Y-%m-%d')}): **${price:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
            st.write(f"   📊 95% CI: ${ci_lower:.2f} - ${ci_upper:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return fitted_arima
        
    except Exception as e:
        st.error(f"❌ Error fitting ARIMA model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None

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
        st.markdown('<h2 class="section-header">📋 Raw Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Data overview tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Quick Stats", "📉 Data Quality"])
        
        with tab1:
            st.dataframe(data, use_container_width=True)
        
        with tab2:
            st.write("**📊 Statistical Summary**")
            st.dataframe(data.describe(), use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📋 Dataset Info**")
                st.write(f"• Rows: {data.shape[0]}")
                st.write(f"• Columns: {data.shape[1]}")
                st.write(f"• Period: {data.index.min().date()} to {data.index.max().date()}")
                st.write(f"• Missing Values: {data.isnull().sum().sum()}")
            
            with col2:
                st.write("**💰 Price Summary**")
                st.write(f"• Highest Price: ${data['High'].max():.2f}")
                st.write(f"• Lowest Price: ${data['Low'].min():.2f}")
                st.write(f"• Average Volume: {data['Volume'].mean():,.0f}")
                st.write(f"• Total Volume: {data['Volume'].sum():,.0f}")
        
        # Data preprocessing
        if st.button("🔧 Preprocess Data", use_container_width=True):
            processed_data = preprocess_data(data.copy())
            st.session_state['processed_data'] = processed_data
        
        # Analysis
        if 'processed_data' in st.session_state:
            processed_data = st.session_state['processed_data']
            
            if st.button("📊 Run Enhanced Analysis", use_container_width=True):
                if analysis_type == "Specific Day":
                    analysis_df = enhanced_analysis(processed_data, analysis_type, analysis_date)
                elif analysis_type == "Specific Week":
                    analysis_df = enhanced_analysis(processed_data, analysis_type, analysis_date)
                else:
                    analysis_df = enhanced_analysis(processed_data, analysis_type, 
                                                 analysis_start=analysis_start, analysis_end=analysis_end)
                
                if analysis_df is not None and not analysis_df.empty:
                    st.session_state['analysis_df'] = analysis_df
            
            # Model prediction section
            st.markdown('<h2 class="section-header">🔮 Machine Learning Predictions</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("🤖 Select Prediction Model", ["LSTM Neural Network", "ARIMA Statistical", "Both Models"])
            with col2:
                st.write("**Model Information:**")
                if model_choice == "LSTM Neural Network":
                    st.info("🧠 Deep learning model for complex pattern recognition")
                elif model_choice == "ARIMA Statistical":
                    st.info("📊 Statistical model for time series forecasting")
                else:
                    st.info("🔄 Compare both models for comprehensive analysis")
            
            if st.button("🚀 Train & Predict", use_container_width=True):
                if model_choice in ["LSTM Neural Network", "Both Models"]:
                    lstm_model, scaler = build_lstm_model(processed_data)
                
                if model_choice in ["ARIMA Statistical", "Both Models"]:
                    arima_model = build_arima_model(processed_data)

if __name__ == "__main__":
    main()
