import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

# TensorFlow/Keras imports with safe configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    # Minimal TensorFlow configuration
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Use CPU only
    tf.config.set_visible_devices([], 'GPU')
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    
    LSTM_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    LSTM_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="📊 Advanced Stock Analysis Platform",
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "NVDA"

# Data preprocessing functions
def check_data_quality(data):
    """Comprehensive data quality analysis"""
    quality_report = {}
    
    # Basic info
    quality_report['total_rows'] = len(data)
    quality_report['date_range'] = (data.index.min(), data.index.max())
    quality_report['columns'] = list(data.columns)
    
    # Missing values
    missing_values = data.isnull().sum()
    quality_report['missing_values'] = missing_values.to_dict()
    quality_report['missing_percentage'] = (missing_values / len(data) * 100).to_dict()
    
    # Duplicates
    quality_report['duplicate_rows'] = data.duplicated().sum()
    
    # Data types
    quality_report['data_types'] = data.dtypes.to_dict()
    
    # Outliers (using IQR method)
    outliers = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    
    quality_report['outliers'] = outliers
    
    # Zero values
    zero_values = (data == 0).sum()
    quality_report['zero_values'] = zero_values.to_dict()
    
    return quality_report

def add_technical_indicators(data):
    """Add technical indicators to the dataset"""
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Bollinger Bands
    bb_window = 20
    df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
    bb_std = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price-based indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Price_Range'] = df['High'] - df['Low']
    df['True_Range'] = np.maximum(df['High'] - df['Low'], 
                                 np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                          abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    
    return df

def clean_data(data):
    """Clean and prepare data"""
    df = data.copy()
    
    # Remove any duplicate dates
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    # Handle missing values
    if df.isnull().any().any():
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove any rows with zero volume (if they exist)
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]
    
    # Ensure positive prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    
    return df

# Header
st.markdown('<h1 class="main-header">📊 Advanced Stock Analysis Platform</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎛️ Control Panel")

# Stock selection
stock_options = {
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation", 
    "META": "Meta Platforms Inc"
}

selected_stock = st.sidebar.selectbox(
    "📈 Select Stock",
    options=list(stock_options.keys()),
    format_func=lambda x: f"{x} - {stock_options[x]}",
    index=0
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "📅 Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )

with col2:
    end_date = st.date_input(
        "📅 End Date", 
        value=datetime.now(),
        max_value=datetime.now()
    )

# Load data button
if st.sidebar.button("📊 Load Stock Data", use_container_width=True):
    try:
        with st.spinner(f"🔄 Loading {selected_stock} data..."):
            ticker = yf.Ticker(selected_stock)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                st.session_state.raw_data = data
                st.session_state.selected_stock = selected_stock
                st.session_state.data_loaded = True
                st.session_state.preprocessing_done = False  # Reset preprocessing
                st.success(f"✅ Successfully loaded {len(data)} days of {selected_stock} data!")
            else:
                st.error("❌ No data found for the selected period.")
                
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

# Main content
if st.session_state.data_loaded:
    data = st.session_state.raw_data
    
    # Current metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
    
    with col2:
        st.metric("Change %", f"{price_change_pct:+.2f}%")
    
    with col3:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
    
    with col4:
        st.metric("Data Points", f"{len(data)} days")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Data Quality", 
        "🛠️ Preprocessing", 
        "📈 Price Analysis", 
        "🔍 ARIMA Analysis",
        "🧠 LSTM Analysis"
    ])
    
    with tab1:
        st.markdown('<h3 class="section-header">🔍 Data Quality Analysis</h3>', unsafe_allow_html=True)
        
        # Run data quality check
        quality_report = check_data_quality(data)
        
        # Display quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", quality_report['total_rows'])
        
        with col2:
            missing_total = sum(quality_report['missing_values'].values())
            st.metric("Missing Values", missing_total)
        
        with col3:
            st.metric("Duplicate Rows", quality_report['duplicate_rows'])
        
        with col4:
            outlier_total = sum(quality_report['outliers'].values())
            st.metric("Total Outliers", outlier_total)
        
        # Date range info
        st.write("**📅 Date Range:**")
        start_date_info, end_date_info = quality_report['date_range']
        st.write(f"From: {start_date_info.strftime('%Y-%m-%d')} to {end_date_info.strftime('%Y-%m-%d')}")
        
        # Missing values analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**❌ Missing Values by Column:**")
            missing_df = pd.DataFrame({
                'Column': list(quality_report['missing_values'].keys()),
                'Missing Count': list(quality_report['missing_values'].values()),
                'Missing %': [f"{x:.2f}%" for x in quality_report['missing_percentage'].values()]
            })
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.write("**🎯 Outliers by Column:**")
            outliers_df = pd.DataFrame({
                'Column': list(quality_report['outliers'].keys()),
                'Outlier Count': list(quality_report['outliers'].values()),
                'Outlier %': [f"{(x/quality_report['total_rows']*100):.2f}%" for x in quality_report['outliers'].values()]
            })
            st.dataframe(outliers_df, use_container_width=True)
        
        # Statistical summary
        st.write("**📈 Statistical Summary:**")
        summary_stats = data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(summary_stats.round(2), use_container_width=True)
    
    with tab2:
        st.markdown('<h3 class="section-header">🛠️ Data Preprocessing</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h4>🔧 Preprocessing Steps:</h4>
        <ul>
            <li><strong>Data Cleaning:</strong> Remove duplicates, handle missing values</li>
            <li><strong>Technical Indicators:</strong> SMA, EMA, MACD, RSI, Bollinger Bands</li>
            <li><strong>Feature Engineering:</strong> Additional analytical features</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Preprocessing", use_container_width=True):
            with st.spinner("🔄 Processing data..."):
                # Clean the data
                cleaned_data = clean_data(data)
                
                # Add technical indicators
                processed_data = add_technical_indicators(cleaned_data)
                
                # Store in session state
                st.session_state.processed_data = processed_data
                st.session_state.preprocessing_done = True
                
                st.success("✅ Preprocessing completed!")
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Before:**")
                    st.write(f"• Rows: {len(data)}")
                    st.write(f"• Columns: {len(data.columns)}")
                    st.write(f"• Missing: {data.isnull().sum().sum()}")
                
                with col2:
                    st.write("**✨ After:**")
                    st.write(f"• Rows: {len(processed_data)}")
                    st.write(f"• Columns: {len(processed_data.columns)}")
                    st.write(f"• Missing: {processed_data.isnull().sum().sum()}")
        
        # Show processed data if available
        if st.session_state.preprocessing_done:
            processed_data = st.session_state.processed_data
            
            st.markdown('<h4>📈 Technical Indicators:</h4>', unsafe_allow_html=True)
            
            # Show sample of processed data
            st.write("**📋 Processed Data (Last 5 Days):**")
            display_cols = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'RSI', 'MACD']
            available_cols = [col for col in display_cols if col in processed_data.columns]
            st.dataframe(processed_data[available_cols].tail(5).round(3), use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="section-header">📈 Price Analysis</h3>', unsafe_allow_html=True)
        
        # Chart controls
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("📊 Chart Type", ["Candlestick", "Line"])
            
        with col2:
            time_period = st.selectbox("📅 Period", ["Last 30 Days", "Last 60 Days", "All Data"])
        
        # Filter data
        if time_period == "Last 30 Days":
            chart_data = data.tail(30)
        elif time_period == "Last 60 Days":
            chart_data = data.tail(60)
        else:
            chart_data = data
        
        # Create chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          subplot_titles=(f'{selected_stock} Price', 'Volume'),
                          row_heights=[0.7, 0.3])
        
        # Add price chart
        if chart_type == "Candlestick":
            price_trace = go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name=selected_stock
            )
        else:  # Line
            price_trace = go.Scatter(
                x=chart_data.index,
                y=chart_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        
        fig.add_trace(price_trace, row=1, col=1)
        
        # Add volume
        fig.add_trace(go.Bar(
            x=chart_data.index,
            y=chart_data['Volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ), row=2, col=1)
        
        fig.update_layout(
            title=f"{selected_stock} - {chart_type} Chart ({time_period})",
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Price Stats:**")
            period_return = ((chart_data['Close'].iloc[-1] / chart_data['Close'].iloc[0]) - 1) * 100
            st.write(f"• Return: {period_return:+.2f}%")
            st.write(f"• High: ${chart_data['High'].max():.2f}")
            st.write(f"• Low: ${chart_data['Low'].min():.2f}")
        
        with col2:
            st.write("**📈 Volume:**")
            avg_volume = chart_data['Volume'].mean()
            st.write(f"• Avg: {avg_volume:,.0f}")
            st.write(f"• Max: {chart_data['Volume'].max():,.0f}")
            st.write(f"• Min: {chart_data['Volume'].min():,.0f}")
        
        with col3:
            st.write("**🎯 Signals:**")
            volatility = chart_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.write(f"• Volatility: {volatility:.2f}%")
            
            if len(chart_data) >= 20:
                sma_20 = chart_data['Close'].rolling(20).mean().iloc[-1]
                trend = "🟢 Bullish" if current_price > sma_20 else "🔴 Bearish"
                st.write(f"• Trend: {trend}")
    
    with tab4:
        st.markdown('<h3 class="section-header">🔍 ARIMA Analysis</h3>', unsafe_allow_html=True)
        
        # Use processed data if available
        analysis_data = st.session_state.processed_data if st.session_state.preprocessing_done else data
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "🎯 Target Variable", 
                ["Close", "SMA_20"] if st.session_state.preprocessing_done else ["Close"]
            )
        
        with col2:
            test_split = st.slider("🔄 Test Split %", 10, 30, 20)
        
        if st.button("🚀 Run ARIMA Analysis", use_container_width=True):
            
            # Prepare data
            if target_column in analysis_data.columns:
                prices = analysis_data[target_column].dropna().values
            else:
                prices = analysis_data['Close'].values
            
            # Stationarity test
            st.write("**🔍 Stationarity Test:**")
            adf_result = adfuller(prices)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"• ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"• p-value: {adf_result[1]:.4f}")
            
            with col2:
                if adf_result[1] <= 0.05:
                    st.success("✅ Data is stationary")
                    use_diff = False
                    arima_data = prices
                else:
                    st.warning("⚠️ Applying differencing")
                    use_diff = True
                    arima_data = np.diff(prices)
            
            # Split data
            split_idx = int(len(arima_data) * (100 - test_split) / 100)
            train_data = arima_data[:split_idx]
            test_data = arima_data[split_idx:]
            
            try:
                # Fit ARIMA model
                with st.spinner("🔄 Training ARIMA..."):
                    model = ARIMA(train_data, order=(1, 1, 1))
                    fitted_model = model.fit()
                
                # Make predictions
                forecast = fitted_model.forecast(steps=len(test_data))
                
                # Calculate metrics
                if use_diff:
                    last_price = prices[split_idx-1]
                    actual_prices = np.cumsum(np.concatenate([[last_price], test_data]))
                    predicted_prices = np.cumsum(np.concatenate([[last_price], forecast]))
                else:
                    actual_prices = test_data
                    predicted_prices = forecast
                
                # Metrics
                mse = mean_squared_error(actual_prices[1:], predicted_prices[1:])
                mae = mean_absolute_error(actual_prices[1:], predicted_prices[1:])
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_prices[1:], predicted_prices[1:])
                mape = np.mean(np.abs((actual_prices[1:] - predicted_prices[1:]) / actual_prices[1:])) * 100
                accuracy = max(0, 100 - mape)
                
                st.success("✅ ARIMA Complete!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col2:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col3:
                    st.metric("MAE", f"${mae:.2f}")
                with col4:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Plot results
                test_dates = analysis_data.index[split_idx:split_idx+len(actual_prices)]
                
                fig_arima = go.Figure()
                
                fig_arima.add_trace(go.Scatter(
                    x=test_dates,
                    y=actual_prices,
                    mode='lines',
                    name=f'Actual {target_column}',
                    line=dict(color='blue', width=2)
                ))
                
                fig_arima.add_trace(go.Scatter(
                    x=test_dates,
                    y=predicted_prices,
                    mode='lines',
                    name='ARIMA Predictions',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_arima.update_layout(
                    title=f"🔍 ARIMA: Actual vs Predicted {target_column}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_arima, use_container_width=True)
                
                # Future predictions
                st.write("**🔮 Next 5 Days:**")
                
                future_forecast = fitted_model.forecast(steps=5)
                
                if use_diff:
                    current_val = analysis_data[target_column].iloc[-1]
                    future_prices = [current_val]
                    for diff_val in future_forecast:
                        future_prices.append(future_prices[-1] + diff_val)
                    future_prices = future_prices[1:]
                else:
                    future_prices = future_forecast
                    current_val = prices[-1]
                
                for i, price in enumerate(future_prices):
                    change = price - current_val
                    change_pct = (change / current_val) * 100
                    direction = "📈" if change > 0 else "📉"
                    st.write(f"{direction} **Day {i+1}**: ${price:.2f} ({change_pct:+.2f}%)")
                
            except Exception as e:
                st.error(f"❌ ARIMA failed: {str(e)}")
    
    with tab5:
        st.markdown('<h3 class="section-header">🧠 LSTM Analysis</h3>', unsafe_allow_html=True)
        
        if not LSTM_AVAILABLE:
            st.error("❌ TensorFlow not available. Install with: pip install tensorflow")
        else:
            st.markdown("""
            <div class="info-card">
            <h4>🧠 LSTM Neural Network:</h4>
            <ul>
                <li><strong>Memory:</strong> Learns long-term patterns</li>
                <li><strong>Non-linear:</strong> Captures complex behaviors</li>
                <li><strong>Robust:</strong> Handles non-stationary data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # LSTM Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                lookback = st.slider("🔄 Lookback Days", 10, 60, 30)
                epochs = st.selectbox("📊 Epochs", [25, 50, 75], index=1)
            
            with col2:
                lstm_units = st.selectbox("🧠 LSTM Units", [32, 50, 64], index=1)
                use_features = st.checkbox("📈 Use Technical Indicators", 
                                         value=st.session_state.preprocessing_done,
                                         disabled=not st.session_state.preprocessing_done)
            
            if st.button("🚀 Train LSTM", use_container_width=True):
                
                # Select features
                if use_features and st.session_state.preprocessing_done:
                    lstm_data = st.session_state.processed_data.copy()
                    feature_cols = ['Close', 'SMA_20', 'RSI']
                    available_features = [col for col in feature_cols if col in lstm_data.columns]
                    if len(available_features) > 1:
                        lstm_features = lstm_data[available_features].fillna(method='ffill')
                        st.info(f"📊 Using: {', '.join(available_features)}")
                    else:
                        lstm_features = lstm_data[['Close']]
                        st.info("📈 Using Close price only")
                else:
                    lstm_features = data[['Close']].copy()
                    st.info("📈 Using Close price only")
                
                # Scale data
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(lstm_features.values)
                
                # Create sequences
                def create_sequences(data, lookback):
                    X, y = [], []
                    for i in range(lookback, len(data)):
                        X.append(data[i-lookback:i])
                        y.append(data[i, 0])  # Predict Close price
                    return np.array(X), np.array(y)
                
                X, y = create_sequences(scaled_data, lookback)
                
                if len(X) < 30:
                    st.error("❌ Not enough data. Try smaller lookback or longer period.")
                else:
                    # Split data
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    st.write(f"**📊 Training: {len(X_train)} | Testing: {len(X_test)}**")
                    
                    try:
                        with st.spinner("🧠 Training LSTM..."):
                            
                            # Build model
                            model = Sequential([
                                LSTM(lstm_units, return_sequences=True, input_shape=(lookback, X_train.shape[2])),
                                Dropout(0.2),
                                LSTM(lstm_units//2),
                                Dropout(0.2),
                                Dense(1)
                            ])
                            
                            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                            
                            # Train
                            history = model.fit(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=32,
                                validation_split=0.1,
                                verbose=0
                            )
                        
                        # Predictions
                        train_pred = model.predict(X_train, verbose=0)
                        test_pred = model.predict(X_test, verbose=0)
                        
                        # Inverse transform (only for Close price column)
                        train_pred_full = np.zeros((len(train_pred), lstm_features.shape[1]))
                        test_pred_full = np.zeros((len(test_pred), lstm_features.shape[1]))
                        train_pred_full[:, 0] = train_pred.flatten()
                        test_pred_full[:, 0] = test_pred.flatten()
                        
                        y_train_full = np.zeros((len(y_train), lstm_features.shape[1]))
                        y_test_full = np.zeros((len(y_test), lstm_features.shape[1]))
                        y_train_full[:, 0] = y_train
                        y_test_full[:, 0] = y_test
                        
                        train_pred_inv = scaler.inverse_transform(train_pred_full)[:, 0]
                        test_pred_inv = scaler.inverse_transform(test_pred_full)[:, 0]
                        y_train_inv = scaler.inverse_transform(y_train_full)[:, 0]
                        y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
                        
                        # Metrics
                        test_mse = mean_squared_error(y_test_inv, test_pred_inv)
                        test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
                        test_rmse = np.sqrt(test_mse)
                        test_r2 = r2_score(y_test_inv, test_pred_inv)
                        test_mape = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
                        test_accuracy = max(0, 100 - test_mape)
                        
                        st.success("✅ LSTM Training Complete!")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{test_accuracy:.2f}%")
                        with col2:
                            st.metric("RMSE", f"${test_rmse:.2f}")
                        with col3:
                            st.metric("MAE", f"${test_mae:.2f}")
                        with col4:
                            st.metric("R² Score", f"{test_r2:.4f}")
                        
                        # Plot predictions
                        train_dates = lstm_features.index[lookback:lookback+len(y_train_inv)]
                        test_dates = lstm_features.index[lookback+len(y_train_inv):lookback+len(y_train_inv)+len(y_test_inv)]
                        
                        fig_lstm = go.Figure()
                        
                        fig_lstm.add_trace(go.Scatter(
                            x=train_dates, y=y_train_inv,
                            mode='lines', name='Actual (Train)',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_lstm.add_trace(go.Scatter(
                            x=test_dates, y=y_test_inv,
                            mode='lines', name='Actual (Test)',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_lstm.add_trace(go.Scatter(
                            x=train_dates, y=train_pred_inv,
                            mode='lines', name='LSTM Pred (Train)',
                            line=dict(color='red', dash='dash', width=1)
                        ))
                        
                        fig_lstm.add_trace(go.Scatter(
                            x=test_dates, y=test_pred_inv,
                            mode='lines', name='LSTM Pred (Test)',
                            line=dict(color='orange', dash='dash', width=2)
                        ))
                        
                        fig_lstm.update_layout(
                            title="🧠 LSTM Predictions vs Actual",
                            xaxis_title="Date", yaxis_title="Price ($)",
                            template="plotly_white", height=500
                        )
                        
                        st.plotly_chart(fig_lstm, use_container_width=True)
                        
                        # Future predictions
                        st.write("**🔮 Next 3 Days:**")
                        
                        last_sequence = scaled_data[-lookback:]
                        future_preds = []
                        
                        current_seq = last_sequence.copy()
                        for _ in range(3):
                            pred = model.predict(current_seq.reshape(1, lookback, -1), verbose=0)
                            future_preds.append(pred[0, 0])
                            
                            # Update sequence
                            new_row = current_seq[-1].copy()
                            new_row[0] = pred[0, 0]
                            current_seq = np.vstack([current_seq[1:], new_row])
                        
                        # Convert to prices
                        future_pred_full = np.zeros((3, lstm_features.shape[1]))
                        future_pred_full[:, 0] = future_preds
                        future_prices = scaler.inverse_transform(future_pred_full)[:, 0]
                        
                        current_actual = lstm_features['Close'].iloc[-1]
                        for i, price in enumerate(future_prices):
                            change = price - current_actual
                            change_pct = (change / current_actual) * 100
                            direction = "📈" if change > 0 else "📉"
                            st.write(f"{direction} **Day {i+1}**: ${price:.2f} ({change_pct:+.2f}%)")
                            
                    except Exception as e:
                        st.error(f"❌ LSTM failed: {str(e)}")

else:
    # Welcome screen
    st.markdown("""
    <div class="info-card">
    <h3>🚀 Welcome to the Advanced Stock Analysis Platform!</h3>
    <p>Comprehensive stock analysis with AI and statistical models.</p>
    <h4>🔥 Enhanced Features:</h4>
    <ul>
        <li><strong>📊 Data Quality Check:</strong> Missing values, outliers, distributions</li>
        <li><strong>🛠️ Preprocessing:</strong> Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)</li>
        <li><strong>📈 Interactive Charts:</strong> Professional candlestick and volume analysis</li>
        <li><strong>🔍 ARIMA Analysis:</strong> Statistical time series with stationarity testing</li>
        <li><strong>🧠 LSTM Networks:</strong> Deep learning predictions with customizable architecture</li>
        <li><strong>🎯 Performance Metrics:</strong> Accuracy, RMSE, MAE, R² for model comparison</li>
        <li><strong>🔮 Future Predictions:</strong> Multi-day forecasts from both models</li>
    </ul>
    <h4>🎓 Quick Start:</h4>
    <ol>
        <li><strong>Select Stock:</strong> Choose NVDA, MSFT, or META</li>
        <li><strong>Load Data:</strong> Get real-time Yahoo Finance data</li>
        <li><strong>Quality Check:</strong> Analyze data quality and distributions</li>
        <li><strong>Preprocessing:</strong> Add technical indicators and features</li>
        <li><strong>Run Analysis:</strong> Compare ARIMA vs LSTM predictions</li>
    </ol>
    <p><strong>👈 Get started by selecting a stock in the sidebar!</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>📊 <strong>Advanced Stock Analysis Platform</strong></p>
    <p>Built with Streamlit • Yahoo Finance • TensorFlow • Statsmodels • Plotly</p>
    <p><em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)