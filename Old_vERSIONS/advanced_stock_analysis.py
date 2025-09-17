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

# Statistical and ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

# TensorFlow/Keras imports with enhanced configuration to prevent mutex blocking
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

try:
    import tensorflow as tf
    
    # Enhanced TensorFlow configuration to prevent mutex blocking
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Disable GPU completely to avoid mutex issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices([], 'GPU')
    
    # Additional mutex prevention settings
    tf.config.experimental.enable_op_determinism()
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    
    LSTM_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    
    # Suppress specific mutex warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
except ImportError:
    LSTM_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. LSTM predictions will be disabled.")
except Exception as e:
    LSTM_AVAILABLE = False
    st.warning(f"⚠️ TensorFlow configuration error: {str(e)}. LSTM predictions will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="🔬 Advanced Stock Analysis Laboratory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #2ca02c;
        margin: 1.5rem 0 0.5rem 0;
        font-weight: bold;
    }
    .analysis-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-card {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTab {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🔬 Advanced Stock Analysis Laboratory</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
<h3>🎯 What This Application Does</h3>
<p>This is a comprehensive stock analysis platform that demonstrates:</p>
<ul>
    <li><strong>📊 Data Preprocessing</strong>: Complete data cleaning and validation</li>
    <li><strong>📈 ARIMA Analysis</strong>: Shows why ARIMA fails on raw prices and succeeds with transformations</li>
    <li><strong>🧠 LSTM Neural Networks</strong>: Deep learning for complex pattern recognition</li>
    <li><strong>📋 Interactive Analysis</strong>: Comprehensive post-preprocessing insights</li>
    <li><strong>🔍 Model Comparison</strong>: Performance analysis across different approaches</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Global variables for session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False

# Sidebar Configuration
st.sidebar.header("🔧 Configuration Panel")

# Stock selection with detailed info
stock_options = {
    "NVIDIA (NVDA)": {
        "symbol": "NVDA",
        "description": "Leading AI/GPU manufacturer",
        "sector": "Technology"
    },
    "Microsoft (MSFT)": {
        "symbol": "MSFT", 
        "description": "Cloud computing and AI (OpenAI partner)",
        "sector": "Technology"
    },
    "Meta (META)": {
        "symbol": "META",
        "description": "Social media and metaverse",
        "sector": "Technology"
    }
}

selected_stock_name = st.sidebar.selectbox(
    "📈 Select Stock for Analysis", 
    list(stock_options.keys()),
    help="Choose the stock you want to analyze"
)

selected_stock = stock_options[selected_stock_name]
st.sidebar.info(f"**Sector**: {selected_stock['sector']}\n\n**Description**: {selected_stock['description']}")

# Date range selection
st.sidebar.subheader("📅 Data Collection Period")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date", 
        datetime.now() - timedelta(days=365),
        help="Select the start date for data collection"
    )

with col2:
    end_date = st.date_input(
        "End Date", 
        datetime.now(),
        help="Select the end date for data collection"
    )

# Data loading function with enhanced error handling
@st.cache_data
def load_stock_data(symbol, start, end):
    """Load stock data with comprehensive error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        
        if data.empty:
            st.error(f"❌ No data available for {symbol} in the specified date range")
            return None
            
        # Add basic technical indicators
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['Price_Range'] = data['High'] - data['Low']
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error loading data for {symbol}: {str(e)}")
        return None

# Data loading section
st.markdown('<h2 class="section-header">📊 Data Collection & Initial Analysis</h2>', unsafe_allow_html=True)

if st.sidebar.button("🔄 Load Stock Data", use_container_width=True):
    with st.spinner(f"📥 Loading data for {selected_stock['symbol']}..."):
        raw_data = load_stock_data(selected_stock['symbol'], start_date, end_date)
    
    if raw_data is not None:
        st.session_state.raw_data = raw_data
        st.session_state.data_loaded = True
        st.session_state.stock_info = selected_stock
        
        st.success(f"✅ Successfully loaded {len(raw_data)} records for {selected_stock_name}")
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(raw_data))
        with col2:
            st.metric("📅 Date Range", f"{(raw_data.index[-1] - raw_data.index[0]).days} days")
        with col3:
            st.metric("💰 Latest Price", f"${raw_data['Close'].iloc[-1]:.2f}")
        with col4:
            latest_return = raw_data['Returns'].iloc[-1] * 100 if not pd.isna(raw_data['Returns'].iloc[-1]) else 0
            st.metric("📈 Last Day Return", f"{latest_return:.2f}%")

# Display loaded data if available
if st.session_state.data_loaded:
    raw_data = st.session_state.raw_data
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">📋 Raw Data Overview</h3>', unsafe_allow_html=True)
    
    # Data quality assessment
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 Price Chart", "🔍 Data Quality", "📉 Returns Analysis"])
    
    with tab1:
        st.write("**First 10 rows of raw data:**")
        st.dataframe(raw_data.head(10), use_container_width=True)
        
        st.write("**Statistical Summary:**")
        st.dataframe(raw_data.describe(), use_container_width=True)
    
    with tab2:
        # Interactive price chart
        fig_price = go.Figure()
        
        fig_price.add_trace(go.Candlestick(
            x=raw_data.index,
            open=raw_data['Open'],
            high=raw_data['High'],
            low=raw_data['Low'],
            close=raw_data['Close'],
            name="Price"
        ))
        
        fig_price.update_layout(
            title=f"📈 {selected_stock_name} - Raw Price Data",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
    
    with tab3:
        # Data quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔍 Data Quality Metrics:**")
            missing_values = raw_data.isnull().sum()
            st.write(f"• Missing values: {missing_values.sum()}")
            st.write(f"• Duplicate rows: {raw_data.duplicated().sum()}")
            st.write(f"• Data types: {len(raw_data.dtypes.unique())} unique types")
            
            # Price validation
            negative_prices = (raw_data[['Open', 'High', 'Low', 'Close']] < 0).any().any()
            st.write(f"• Negative prices: {'❌ Found' if negative_prices else '✅ None'}")
            
            # Logical price relationships
            price_logic = ((raw_data['High'] >= raw_data['Low']) & 
                          (raw_data['High'] >= raw_data['Open']) & 
                          (raw_data['High'] >= raw_data['Close']) &
                          (raw_data['Low'] <= raw_data['Open']) & 
                          (raw_data['Low'] <= raw_data['Close'])).all()
            st.write(f"• Price logic: {'✅ Valid' if price_logic else '❌ Invalid'}")
        
        with col2:
            # Data distribution visualization
            fig_dist = go.Figure()
            
            for col in ['Open', 'High', 'Low', 'Close']:
                fig_dist.add_trace(go.Box(
                    y=raw_data[col],
                    name=col,
                    boxpoints='outliers'
                ))
            
            fig_dist.update_layout(
                title="📊 Price Distribution (Box Plot)",
                yaxis_title="Price ($)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        # Returns analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns histogram
            fig_returns = px.histogram(
                raw_data.dropna(),
                x='Returns',
                nbins=50,
                title="📊 Daily Returns Distribution",
                labels={'Returns': 'Daily Returns', 'count': 'Frequency'}
            )
            fig_returns.update_layout(template="plotly_white")
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Volatility over time
            fig_vol = px.line(
                raw_data.dropna(),
                x=raw_data.dropna().index,
                y='Volatility',
                title="📈 Rolling Volatility (20-day)",
                labels={'Volatility': 'Volatility', 'x': 'Date'}
            )
            fig_vol.update_layout(template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Data Preprocessing Section
def comprehensive_preprocessing(data):
    """Comprehensive data preprocessing with visualizations"""
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing & Transformation</h2>', unsafe_allow_html=True)
    
    processed_data = data.copy()
    
    # Step 1: Handle missing values
    st.markdown('<h3 class="subsection-header">1️⃣ Missing Value Treatment</h3>', unsafe_allow_html=True)
    
    missing_before = processed_data.isnull().sum().sum()
    if missing_before > 0:
        st.warning(f"⚠️ Found {missing_before} missing values. Applying forward fill...")
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        st.success("✅ Missing values handled using forward/backward fill")
    else:
        st.success("✅ No missing values found!")
    
    # Step 2: Remove duplicates
    st.markdown('<h3 class="subsection-header">2️⃣ Duplicate Removal</h3>', unsafe_allow_html=True)
    
    duplicates_before = processed_data.duplicated().sum()
    if duplicates_before > 0:
        processed_data = processed_data.drop_duplicates()
        st.warning(f"⚠️ Removed {duplicates_before} duplicate rows")
    else:
        st.success("✅ No duplicate rows found!")
    
    # Step 3: Create additional features
    st.markdown('<h3 class="subsection-header">3️⃣ Feature Engineering</h3>', unsafe_allow_html=True)
    
    # Technical indicators
    processed_data['SMA_5'] = processed_data['Close'].rolling(window=5).mean()
    processed_data['SMA_20'] = processed_data['Close'].rolling(window=20).mean()
    processed_data['EMA_12'] = processed_data['Close'].ewm(span=12).mean()
    processed_data['EMA_26'] = processed_data['Close'].ewm(span=26).mean()
    processed_data['MACD'] = processed_data['EMA_12'] - processed_data['EMA_26']
    
    # Bollinger Bands
    processed_data['BB_Middle'] = processed_data['Close'].rolling(window=20).mean()
    processed_data['BB_Std'] = processed_data['Close'].rolling(window=20).std()
    processed_data['BB_Upper'] = processed_data['BB_Middle'] + (processed_data['BB_Std'] * 2)
    processed_data['BB_Lower'] = processed_data['BB_Middle'] - (processed_data['BB_Std'] * 2)
    
    # RSI
    delta = processed_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    processed_data['RSI'] = 100 - (100 / (1 + rs))
    
    st.success("✅ Created technical indicators: SMA, EMA, MACD, Bollinger Bands, RSI")
    
    # Visualization of preprocessing effects
    col1, col2 = st.columns(2)
    
    with col1:
        # Before preprocessing
        fig_before = go.Figure()
        fig_before.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Raw Price',
            line=dict(color='red', width=2)
        ))
        fig_before.update_layout(
            title="📉 Before Preprocessing",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_before, use_container_width=True)
    
    with col2:
        # After preprocessing with technical indicators
        fig_after = go.Figure()
        fig_after.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        fig_after.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['SMA_20'],
            mode='lines',
            name='SMA-20',
            line=dict(color='orange', width=1)
        ))
        fig_after.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig_after.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
        fig_after.update_layout(
            title="📈 After Preprocessing",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_after, use_container_width=True)
    
    return processed_data

# ARIMA Analysis Section - Showing Transformation Importance
def arima_analysis_comprehensive(data):
    """Comprehensive ARIMA analysis showing why transformations matter"""
    st.markdown('<h2 class="section-header">📊 ARIMA Analysis: The Power of Transformations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h4>🎯 What We'll Demonstrate:</h4>
    <ul>
        <li><strong>Why ARIMA fails on raw prices:</strong> Non-stationarity issues</li>
        <li><strong>How transformations help:</strong> Differencing and log transforms</li>
        <li><strong>Model performance comparison:</strong> Raw vs Transformed data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    prices = data['Close'].dropna()
    
    # Step 1: Test stationarity on raw prices
    st.markdown('<h3 class="subsection-header">1️⃣ Stationarity Testing on Raw Prices</h3>', unsafe_allow_html=True)
    
    def test_stationarity(timeseries, title):
        # ADF Test
        adf_result = adfuller(timeseries)
        
        # KPSS Test
        kpss_result = kpss(timeseries, regression='c')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ADF Test Results for {title}:**")
            st.write(f"• ADF Statistic: {adf_result[0]:.6f}")
            st.write(f"• p-value: {adf_result[1]:.6f}")
            st.write(f"• Critical Values:")
            for key, value in adf_result[4].items():
                st.write(f"  - {key}: {value:.6f}")
            
            if adf_result[1] <= 0.05:
                st.success("✅ ADF: Series is stationary")
            else:
                st.error("❌ ADF: Series is NOT stationary")
        
        with col2:
            st.write(f"**KPSS Test Results for {title}:**")
            st.write(f"• KPSS Statistic: {kpss_result[0]:.6f}")
            st.write(f"• p-value: {kpss_result[1]:.6f}")
            st.write(f"• Critical Values:")
            for key, value in kpss_result[3].items():
                st.write(f"  - {key}: {value:.6f}")
            
            if kpss_result[1] >= 0.05:
                st.success("✅ KPSS: Series is stationary")
            else:
                st.error("❌ KPSS: Series is NOT stationary")
        
        return adf_result[1] <= 0.05 and kpss_result[1] >= 0.05
    
    # Test raw prices
    raw_stationary = test_stationarity(prices, "Raw Prices")
    
    # Visualize raw prices
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=data.index,
        y=prices,
        mode='lines',
        name='Raw Prices',
        line=dict(color='red', width=2)
    ))
    fig_raw.update_layout(
        title="📈 Raw Stock Prices (Non-Stationary)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_raw, use_container_width=True)
    
    # Step 2: Apply transformations
    st.markdown('<h3 class="subsection-header">2️⃣ Data Transformations</h3>', unsafe_allow_html=True)
    
    # First difference
    prices_diff = prices.diff().dropna()
    
    # Log transformation then difference
    log_prices = np.log(prices)
    log_prices_diff = log_prices.diff().dropna()
    
    # Test transformations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 First Difference Transform:**")
        diff_stationary = test_stationarity(prices_diff, "First Difference")
        
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=prices_diff.index,
            y=prices_diff,
            mode='lines',
            name='First Difference',
            line=dict(color='blue', width=2)
        ))
        fig_diff.update_layout(
            title="📊 First Difference of Prices",
            xaxis_title="Date",
            yaxis_title="Price Change ($)",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    
    with col2:
        st.write("**📊 Log + First Difference Transform:**")
        log_diff_stationary = test_stationarity(log_prices_diff, "Log + First Difference")
        
        fig_log_diff = go.Figure()
        fig_log_diff.add_trace(go.Scatter(
            x=log_prices_diff.index,
            y=log_prices_diff,
            mode='lines',
            name='Log + First Difference',
            line=dict(color='green', width=2)
        ))
        fig_log_diff.update_layout(
            title="📊 Log + First Difference",
            xaxis_title="Date",
            yaxis_title="Log Price Change",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_log_diff, use_container_width=True)
    
    # Step 3: ARIMA modeling comparison
    st.markdown('<h3 class="subsection-header">3️⃣ ARIMA Model Performance Comparison</h3>', unsafe_allow_html=True)
    
    def fit_arima_and_evaluate(data_series, title, order=(1,1,1)):
        try:
            # Split data
            train_size = int(len(data_series) * 0.8)
            train, test = data_series[:train_size], data_series[train_size:]
            
            # Fit ARIMA model
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            
            # Calculate metrics
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mse)
            
            # Calculate percentage accuracy
            mape = np.mean(np.abs((test - forecast) / test)) * 100
            accuracy = 100 - mape
            
            return {
                'model': fitted_model,
                'forecast': forecast,
                'test': test,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        except Exception as e:
            st.error(f"❌ Failed to fit ARIMA for {title}: {str(e)}")
            return None
    
    # Compare different approaches
    st.write("**🔄 Training ARIMA models on different data transformations...**")
    
    models_comparison = {}
    
    # Only proceed if we have enough data
    if len(prices) > 50:
        # Model 1: Raw prices (will likely fail or perform poorly)
        if not raw_stationary:
            st.warning("⚠️ Attempting ARIMA on non-stationary raw prices (expected to perform poorly)")
        
        raw_result = fit_arima_and_evaluate(prices, "Raw Prices", order=(1,1,1))
        if raw_result:
            models_comparison['Raw Prices'] = raw_result
        
        # Model 2: First difference
        diff_result = fit_arima_and_evaluate(prices_diff, "First Difference", order=(1,0,1))
        if diff_result:
            models_comparison['First Difference'] = diff_result
        
        # Model 3: Log + First difference
        log_diff_result = fit_arima_and_evaluate(log_prices_diff, "Log + First Difference", order=(1,0,1))
        if log_diff_result:
            models_comparison['Log + First Difference'] = log_diff_result
        
        # Display results comparison
        if models_comparison:
            st.markdown('<h4>📊 Model Performance Comparison</h4>', unsafe_allow_html=True)
            
            # Create comparison table
            comparison_data = []
            for name, result in models_comparison.items():
                comparison_data.append({
                    'Model': name,
                    'MSE': f"{result['mse']:.6f}",
                    'MAE': f"{result['mae']:.6f}",
                    'RMSE': f"{result['rmse']:.6f}",
                    'Accuracy (%)': f"{result['accuracy']:.2f}%",
                    'AIC': f"{result['aic']:.2f}",
                    'BIC': f"{result['bic']:.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization of forecasts
            fig_comparison = go.Figure()
            
            for name, result in models_comparison.items():
                if name == 'Raw Prices':
                    # For raw prices, show actual forecast
                    fig_comparison.add_trace(go.Scatter(
                        x=result['test'].index,
                        y=result['test'],
                        mode='lines',
                        name=f'Actual {name}',
                        line=dict(width=3)
                    ))
                    fig_comparison.add_trace(go.Scatter(
                        x=result['test'].index,
                        y=result['forecast'],
                        mode='lines',
                        name=f'ARIMA Forecast {name}',
                        line=dict(dash='dash', width=2)
                    ))
                else:
                    # For differences, we need to transform back to price level
                    if name == 'First Difference':
                        # Cumulative sum to get back to price level
                        actual_prices = result['test'].cumsum() + prices.iloc[int(len(prices) * 0.8) - 1]
                        forecast_prices = result['forecast'].cumsum() + prices.iloc[int(len(prices) * 0.8) - 1]
                    else:  # Log + First Difference
                        # Transform back from log space
                        log_level = log_prices.iloc[int(len(log_prices) * 0.8) - 1]
                        actual_log_prices = result['test'].cumsum() + log_level
                        forecast_log_prices = result['forecast'].cumsum() + log_level
                        actual_prices = np.exp(actual_log_prices)
                        forecast_prices = np.exp(forecast_log_prices)
                    
                    fig_comparison.add_trace(go.Scatter(
                        x=result['test'].index,
                        y=actual_prices,
                        mode='lines',
                        name=f'Actual (from {name})',
                        line=dict(width=2)
                    ))
                    fig_comparison.add_trace(go.Scatter(
                        x=result['test'].index,
                        y=forecast_prices,
                        mode='lines',
                        name=f'ARIMA Forecast (from {name})',
                        line=dict(dash='dash', width=2)
                    ))
            
            fig_comparison.update_layout(
                title="📊 ARIMA Forecasts Comparison: Raw vs Transformed Data",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Key insights
            if len(models_comparison) > 1:
                best_model = min(models_comparison.items(), key=lambda x: x[1]['mse'])
                worst_model = max(models_comparison.items(), key=lambda x: x[1]['mse'])
                
                st.markdown(f"""
                <div class="success-card">
                <h4>🎯 Key Insights:</h4>
                <ul>
                    <li><strong>Best performing model:</strong> {best_model[0]} (MSE: {best_model[1]['mse']:.6f})</li>
                    <li><strong>Worst performing model:</strong> {worst_model[0]} (MSE: {worst_model[1]['mse']:.6f})</li>
                    <li><strong>Performance improvement:</strong> {((worst_model[1]['mse'] - best_model[1]['mse']) / worst_model[1]['mse'] * 100):.1f}% better MSE</li>
                </ul>
                <p><strong>Conclusion:</strong> Data transformation is crucial for ARIMA model performance!</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Not enough data points for reliable ARIMA analysis. Please select a longer date range.")
    
    return models_comparison

if st.session_state.data_loaded and st.button("🔧 Start Preprocessing", use_container_width=True):
    processed_data = comprehensive_preprocessing(st.session_state.raw_data)
    st.session_state.processed_data = processed_data
    st.session_state.data_preprocessed = True

# ARIMA Analysis
if st.session_state.data_loaded and st.button("📊 Run ARIMA Analysis", use_container_width=True):
    arima_results = arima_analysis_comprehensive(st.session_state.raw_data)
    st.session_state.arima_results = arima_results

# LSTM Analysis Section
def lstm_analysis_comprehensive(data):
    """Comprehensive LSTM analysis with proper sequence generation"""
    if not LSTM_AVAILABLE:
        st.error("❌ TensorFlow/Keras not available. LSTM analysis cannot be performed.")
        return None
    
    st.markdown('<h2 class="section-header">🧠 LSTM Neural Network Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h4>🎯 LSTM Advantages:</h4>
    <ul>
        <li><strong>Memory:</strong> Can remember long-term dependencies in sequential data</li>
        <li><strong>Non-linearity:</strong> Captures complex patterns that ARIMA cannot</li>
        <li><strong>Robustness:</strong> Works well with non-stationary data without transformation</li>
        <li><strong>Feature learning:</strong> Automatically learns relevant features</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Data preparation
    st.markdown('<h3 class="subsection-header">1️⃣ Data Preparation for LSTM</h3>', unsafe_allow_html=True)
    
    # Use closing prices
    prices = data['Close'].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))
    
    # Parameters
    lookback_window = st.sidebar.slider("🔄 LSTM Lookback Window", 10, 100, 60, help="Number of previous days to use for prediction")
    
    def create_sequences(data, lookback):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    # Create sequences
    X, y = create_sequences(scaled_data, lookback_window)
    
    if len(X) < 20:
        st.error("❌ Not enough data for LSTM training. Please select a longer date range.")
        return None
    
    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Data Preparation Summary:**")
        st.write(f"• Total sequences: {len(X)}")
        st.write(f"• Training sequences: {len(X_train)}")
        st.write(f"• Testing sequences: {len(X_test)}")
        st.write(f"• Lookback window: {lookback_window} days")
        st.write(f"• Features per sequence: {X_train.shape[2]}")
    
    with col2:
        # Visualize sequence example
        if len(X_train) > 0:
            example_seq = X_train[0].flatten()
            example_target = y_train[0]
            
            fig_seq = go.Figure()
            fig_seq.add_trace(go.Scatter(
                x=list(range(len(example_seq))),
                y=example_seq,
                mode='lines+markers',
                name='Input Sequence',
                line=dict(color='blue', width=2)
            ))
            fig_seq.add_trace(go.Scatter(
                x=[len(example_seq)],
                y=[example_target],
                mode='markers',
                name='Target (Next Day)',
                marker=dict(color='red', size=10)
            ))
            fig_seq.update_layout(
                title="📈 Example LSTM Input Sequence",
                xaxis_title="Days (Lookback)",
                yaxis_title="Scaled Price",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_seq, use_container_width=True)
    
    # Model building
    st.markdown('<h3 class="subsection-header">2️⃣ LSTM Model Architecture</h3>', unsafe_allow_html=True)
    
    # Model parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lstm_units_1 = st.selectbox("🧠 LSTM Units (Layer 1)", [32, 50, 64, 100], index=1)
        lstm_units_2 = st.selectbox("🧠 LSTM Units (Layer 2)", [25, 32, 50, 64], index=0)
    
    with col2:
        dropout_rate = st.selectbox("🎯 Dropout Rate", [0.1, 0.2, 0.3, 0.4], index=1)
        batch_size = st.selectbox("📦 Batch Size", [16, 32, 64], index=1)
    
    with col3:
        epochs = st.selectbox("🔄 Training Epochs", [25, 50, 75, 100], index=1)
        patience = st.selectbox("⏰ Early Stopping Patience", [5, 10, 15], index=1)
    
    # Build LSTM model
    def build_lstm_model(input_shape):
        model = Sequential([
            LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units_2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    # Display model architecture
    st.write("**🏗️ Model Architecture:**")
    st.code(f"""
    Sequential([
        LSTM({lstm_units_1}, return_sequences=True, input_shape=({lookback_window}, 1)),
        Dropout({dropout_rate}),
        LSTM({lstm_units_2}, return_sequences=False),
        Dropout({dropout_rate}),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    Optimizer: Adam (lr=0.001)
    Loss: Mean Squared Error
    Metrics: Mean Absolute Error
    """)
    
    # Model training
    st.markdown('<h3 class="subsection-header">3️⃣ Model Training & Evaluation</h3>', unsafe_allow_html=True)
    
    if st.button("🚀 Train LSTM Model", use_container_width=True):
        
        with st.spinner("🧠 Training LSTM model... This may take a few minutes."):
            # Build model
            model = build_lstm_model((lookback_window, 1))
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            
            # Train model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.6f} - Val Loss: {logs['val_loss']:.6f}")
            
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                callbacks=[early_stopping, StreamlitCallback()],
                verbose=0
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")
        
        # Model evaluation
        st.markdown('<h3 class="subsection-header">4️⃣ Model Performance Analysis</h3>', unsafe_allow_html=True)
        
        # Make predictions
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_actual, train_predict)
        test_mse = mean_squared_error(y_test_actual, test_predict)
        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train_actual, train_predict)
        test_r2 = r2_score(y_test_actual, test_predict)
        
        # Calculate accuracy (1 - MAPE)
        train_mape = np.mean(np.abs((y_train_actual - train_predict) / y_train_actual)) * 100
        test_mape = np.mean(np.abs((y_test_actual - test_predict) / y_test_actual)) * 100
        train_accuracy = 100 - train_mape
        test_accuracy = 100 - test_mape
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Training Performance:**")
            st.metric("MSE", f"{train_mse:.2f}")
            st.metric("MAE", f"${train_mae:.2f}")
            st.metric("RMSE", f"${train_rmse:.2f}")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("Accuracy", f"{train_accuracy:.2f}%")
        
        with col2:
            st.write("**🎯 Testing Performance:**")
            st.metric("MSE", f"{test_mse:.2f}")
            st.metric("MAE", f"${test_mae:.2f}")
            st.metric("RMSE", f"${test_rmse:.2f}")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("Accuracy", f"{test_accuracy:.2f}%")
        
        # Training history visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curves
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(history.history['loss']) + 1)),
                y=history.history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(history.history['val_loss']) + 1)),
                y=history.history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            fig_loss.update_layout(
                title="📉 Training & Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            # MAE curves
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=list(range(1, len(history.history['mae']) + 1)),
                y=history.history['mae'],
                mode='lines',
                name='Training MAE',
                line=dict(color='blue', width=2)
            ))
            fig_mae.add_trace(go.Scatter(
                x=list(range(1, len(history.history['val_mae']) + 1)),
                y=history.history['val_mae'],
                mode='lines',
                name='Validation MAE',
                line=dict(color='red', width=2)
            ))
            fig_mae.update_layout(
                title="📊 Training & Validation MAE",
                xaxis_title="Epoch",
                yaxis_title="Mean Absolute Error",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Predictions visualization
        st.markdown('<h3 class="subsection-header">5️⃣ Predictions Visualization</h3>', unsafe_allow_html=True)
        
        # Create dates for plotting
        train_dates = data.index[lookback_window:lookback_window+len(y_train_actual)]
        test_dates = data.index[lookback_window+len(y_train_actual):lookback_window+len(y_train_actual)+len(y_test_actual)]
        
        fig_pred = go.Figure()
        
        # Actual prices
        fig_pred.add_trace(go.Scatter(
            x=train_dates,
            y=y_train_actual.flatten(),
            mode='lines',
            name='Actual (Training)',
            line=dict(color='blue', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual (Testing)',
            line=dict(color='green', width=2)
        ))
        
        # Predicted prices
        fig_pred.add_trace(go.Scatter(
            x=train_dates,
            y=train_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Training)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Testing)',
            line=dict(color='orange', dash='dash', width=2)
        ))
        
        fig_pred.update_layout(
            title="🧠 LSTM Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Future predictions
        st.markdown('<h3 class="subsection-header">6️⃣ Future Predictions</h3>', unsafe_allow_html=True)
        
        # Predict next 5 days
        last_sequence = scaled_data[-lookback_window:]
        future_predictions = []
        
        for _ in range(5):
            next_pred = model.predict(last_sequence.reshape(1, lookback_window, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            # Update sequence
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔮 LSTM Future Predictions:**")
            current_price = data['Close'].iloc[-1]
            
            for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
                change = price[0] - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉"
                
                st.write(f"{direction} **Day {i+1}** ({date.strftime('%Y-%m-%d')}): **${price[0]:.2f}**")
                st.write(f"   Change: ${change:.2f} ({change_pct:+.2f}%)")
        
        with col2:
            # Future predictions chart
            fig_future = go.Figure()
            
            # Last 20 days of actual data
            recent_data = data['Close'].tail(20)
            fig_future.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data.values,
                mode='lines+markers',
                name='Recent Actual Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Future predictions
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions.flatten(),
                mode='lines+markers',
                name='LSTM Future Predictions',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig_future.update_layout(
                title="🔮 LSTM Future Predictions",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
        
        # Store results
        lstm_results = {
            'model': model,
            'scaler': scaler,
            'train_metrics': {
                'mse': train_mse,
                'mae': train_mae,
                'rmse': train_rmse,
                'r2': train_r2,
                'accuracy': train_accuracy
            },
            'test_metrics': {
                'mse': test_mse,
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'accuracy': test_accuracy
            },
            'future_predictions': future_predictions,
            'future_dates': future_dates,
            'history': history.history
        }
        
        return lstm_results
    
    return None

# Interactive Analysis Section
def interactive_analysis_comprehensive():
    """Comprehensive interactive analysis and model comparison"""
    st.markdown('<h2 class="section-header">🎯 Interactive Analysis & Model Comparison</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first before running analysis.")
        return
    
    # Model comparison overview
    st.markdown("""
    <div class="info-card">
    <h4>📊 Model Comparison Overview:</h4>
    <ul>
        <li><strong>ARIMA:</strong> Statistical model for linear trends and seasonality</li>
        <li><strong>LSTM:</strong> Neural network for complex non-linear patterns</li>
        <li><strong>Combined Insights:</strong> Comprehensive understanding of stock behavior</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if analyses have been run
    arima_available = hasattr(st.session_state, 'arima_results') and st.session_state.arima_results is not None
    lstm_available = hasattr(st.session_state, 'lstm_results') and st.session_state.lstm_results is not None
    
    if not arima_available and not lstm_available:
        st.warning("⚠️ Please run ARIMA and/or LSTM analysis first to view interactive comparisons.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Quick ARIMA Analysis", use_container_width=True):
                arima_results = arima_analysis_comprehensive(st.session_state.raw_data)
                st.session_state.arima_results = arima_results
                st.rerun()
        
        with col2:
            if st.button("🧠 Quick LSTM Analysis", use_container_width=True):
                lstm_results = lstm_analysis_comprehensive(st.session_state.raw_data)
                if lstm_results:
                    st.session_state.lstm_results = lstm_results
                st.rerun()
        
        return
    
    # Analysis tabs
    analysis_tabs = st.tabs([
        "📊 Model Performance Comparison", 
        "🔮 Future Predictions Comparison",
        "📈 Interactive Price Analysis", 
        "🎯 Trading Insights",
        "📋 Summary Report"
    ])
    
    # Tab 1: Model Performance Comparison
    with analysis_tabs[0]:
        st.markdown('<h3 class="subsection-header">🏆 Model Performance Metrics</h3>', unsafe_allow_html=True)
        
        if arima_available and lstm_available:
            # Create comparison table
            comparison_data = {
                'Metric': ['Accuracy (%)', 'RMSE ($)', 'MAE ($)', 'R² Score', 'Model Type', 'Best For'],
                'ARIMA': [
                    f"{getattr(st.session_state.arima_results.get('best_accuracy', {}), 'accuracy', 0):.2f}%" if arima_available else "N/A",
                    f"${getattr(st.session_state.arima_results.get('best_metrics', {}), 'rmse', 0):.2f}" if arima_available else "N/A",
                    f"${getattr(st.session_state.arima_results.get('best_metrics', {}), 'mae', 0):.2f}" if arima_available else "N/A",
                    f"{getattr(st.session_state.arima_results.get('best_metrics', {}), 'r2', 0):.4f}" if arima_available else "N/A",
                    "Statistical",
                    "Linear trends, seasonality"
                ],
                'LSTM': [
                    f"{st.session_state.lstm_results['test_metrics']['accuracy']:.2f}%" if lstm_available else "N/A",
                    f"${st.session_state.lstm_results['test_metrics']['rmse']:.2f}" if lstm_available else "N/A",
                    f"${st.session_state.lstm_results['test_metrics']['mae']:.2f}" if lstm_available else "N/A",
                    f"{st.session_state.lstm_results['test_metrics']['r2']:.4f}" if lstm_available else "N/A",
                    "Neural Network",
                    "Complex patterns, non-linearity"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Style the comparison table
            st.markdown("**📊 Comprehensive Model Comparison:**")
            st.table(comparison_df.set_index('Metric'))
            
            # Performance visualization
            if arima_available and lstm_available:
                metrics = ['Accuracy (%)', 'R² Score']
                arima_vals = [
                    getattr(st.session_state.arima_results.get('best_accuracy', {}), 'accuracy', 0),
                    getattr(st.session_state.arima_results.get('best_metrics', {}), 'r2', 0) * 100
                ]
                lstm_vals = [
                    st.session_state.lstm_results['test_metrics']['accuracy'],
                    st.session_state.lstm_results['test_metrics']['r2'] * 100
                ]
                
                fig_comparison = go.Figure(data=[
                    go.Bar(name='ARIMA', x=metrics, y=arima_vals, marker_color='blue'),
                    go.Bar(name='LSTM', x=metrics, y=lstm_vals, marker_color='red')
                ])
                
                fig_comparison.update_layout(
                    title="🏆 Model Performance Comparison",
                    xaxis_title="Metrics",
                    yaxis_title="Score",
                    barmode='group',
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        elif arima_available:
            st.info("📊 Only ARIMA results available. Run LSTM analysis for comparison.")
            
            arima_metrics = st.session_state.arima_results.get('best_metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ARIMA Accuracy", f"{getattr(st.session_state.arima_results.get('best_accuracy', {}), 'accuracy', 0):.2f}%")
            with col2:
                st.metric("RMSE", f"${arima_metrics.get('rmse', 0):.2f}")
            with col3:
                st.metric("MAE", f"${arima_metrics.get('mae', 0):.2f}")
            with col4:
                st.metric("R² Score", f"{arima_metrics.get('r2', 0):.4f}")
        
        elif lstm_available:
            st.info("🧠 Only LSTM results available. Run ARIMA analysis for comparison.")
            
            lstm_metrics = st.session_state.lstm_results['test_metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("LSTM Accuracy", f"{lstm_metrics['accuracy']:.2f}%")
            with col2:
                st.metric("RMSE", f"${lstm_metrics['rmse']:.2f}")
            with col3:
                st.metric("MAE", f"${lstm_metrics['mae']:.2f}")
            with col4:
                st.metric("R² Score", f"{lstm_metrics['r2']:.4f}")
    
    # Tab 2: Future Predictions Comparison
    with analysis_tabs[1]:
        st.markdown('<h3 class="subsection-header">🔮 Future Predictions Analysis</h3>', unsafe_allow_html=True)
        
        if arima_available or lstm_available:
            # Predict next 7 days
            fig_future = go.Figure()
            
            # Recent actual data (last 30 days)
            recent_data = st.session_state.raw_data['Close'].tail(30)
            fig_future.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data.values,
                mode='lines+markers',
                name='Recent Actual Prices',
                line=dict(color='blue', width=3)
            ))
            
            current_price = st.session_state.raw_data['Close'].iloc[-1]
            
            if arima_available:
                # ARIMA future predictions (extend to 7 days)
                arima_forecast = st.session_state.arima_results.get('forecast', {})
                if arima_forecast:
                    forecast_values = arima_forecast.get('forecast', [])
                    if len(forecast_values) > 0:
                        last_date = st.session_state.raw_data.index[-1]
                        arima_future_dates = pd.date_range(
                            start=last_date + timedelta(days=1), 
                            periods=min(7, len(forecast_values)), 
                            freq='D'
                        )
                        
                        fig_future.add_trace(go.Scatter(
                            x=arima_future_dates,
                            y=forecast_values[:len(arima_future_dates)],
                            mode='lines+markers',
                            name='ARIMA Predictions',
                            line=dict(color='green', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
            
            if lstm_available:
                # LSTM future predictions
                lstm_future_dates = st.session_state.lstm_results['future_dates']
                lstm_predictions = st.session_state.lstm_results['future_predictions']
                
                fig_future.add_trace(go.Scatter(
                    x=lstm_future_dates,
                    y=lstm_predictions.flatten(),
                    mode='lines+markers',
                    name='LSTM Predictions',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=8)
                ))
            
            fig_future.update_layout(
                title="🔮 Future Predictions Comparison",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Predictions summary
            col1, col2 = st.columns(2)
            
            with col1:
                if arima_available:
                    st.write("**📊 ARIMA Predictions:**")
                    arima_forecast = st.session_state.arima_results.get('forecast', {})
                    if arima_forecast and arima_forecast.get('forecast'):
                        for i, price in enumerate(arima_forecast['forecast'][:5]):
                            change = price - current_price
                            change_pct = (change / current_price) * 100
                            direction = "📈" if change > 0 else "📉"
                            st.write(f"{direction} Day {i+1}: ${price:.2f} ({change_pct:+.2f}%)")
                    else:
                        st.write("No ARIMA forecast available")
            
            with col2:
                if lstm_available:
                    st.write("**🧠 LSTM Predictions:**")
                    lstm_predictions = st.session_state.lstm_results['future_predictions']
                    for i, price in enumerate(lstm_predictions[:5]):
                        change = price[0] - current_price
                        change_pct = (change / current_price) * 100
                        direction = "📈" if change > 0 else "📉"
                        st.write(f"{direction} Day {i+1}: ${price[0]:.2f} ({change_pct:+.2f}%)")
        
        else:
            st.warning("⚠️ Run model analysis first to view future predictions.")
    
    # Tab 3: Interactive Price Analysis
    with analysis_tabs[2]:
        st.markdown('<h3 class="subsection-header">📈 Interactive Price Movement Analysis</h3>', unsafe_allow_html=True)
        
        # Price analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_period = st.selectbox(
                "📅 Analysis Period", 
                ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Full Period"],
                index=1
            )
        
        with col2:
            show_volume = st.checkbox("📊 Show Volume", value=True)
        
        with col3:
            show_indicators = st.checkbox("📈 Show Technical Indicators", value=True)
        
        # Filter data based on period
        if analysis_period == "Last 30 Days":
            plot_data = st.session_state.raw_data.tail(30)
        elif analysis_period == "Last 60 Days":
            plot_data = st.session_state.raw_data.tail(60)
        elif analysis_period == "Last 90 Days":
            plot_data = st.session_state.raw_data.tail(90)
        else:
            plot_data = st.session_state.raw_data
        
        # Create subplots
        if show_volume:
            fig_interactive = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price & Technical Indicators', 'Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig_interactive = go.Figure()
        
        # Candlestick chart
        fig_interactive.add_trace(
            go.Candlestick(
                x=plot_data.index,
                open=plot_data['Open'],
                high=plot_data['High'],
                low=plot_data['Low'],
                close=plot_data['Close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Technical indicators
        if show_indicators and hasattr(st.session_state, 'processed_data'):
            processed_data = st.session_state.processed_data
            
            # Moving averages
            if 'SMA_20' in processed_data.columns:
                fig_interactive.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=processed_data.loc[plot_data.index, 'SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'EMA_20' in processed_data.columns:
                fig_interactive.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=processed_data.loc[plot_data.index, 'EMA_20'],
                        mode='lines',
                        name='EMA 20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'BB_Upper' in processed_data.columns:
                fig_interactive.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=processed_data.loc[plot_data.index, 'BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                fig_interactive.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=processed_data.loc[plot_data.index, 'BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Volume
        if show_volume:
            fig_interactive.add_trace(
                go.Bar(
                    x=plot_data.index,
                    y=plot_data['Volume'],
                    name="Volume",
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig_interactive.update_layout(
            title=f"📈 Interactive Price Analysis - {st.session_state.selected_stock}",
            template="plotly_white",
            height=700 if show_volume else 500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        if show_volume:
            fig_interactive.update_xaxes(title_text="Date", row=2, col=1)
            fig_interactive.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_interactive.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig_interactive.update_xaxes(title_text="Date")
            fig_interactive.update_yaxes(title_text="Price ($)")
        
        st.plotly_chart(fig_interactive, use_container_width=True)
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = plot_data['Close'].iloc[-1] - plot_data['Close'].iloc[0]
            price_change_pct = (price_change / plot_data['Close'].iloc[0]) * 100
            st.metric("Period Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
        
        with col2:
            volatility = plot_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Volatility (Annualized)", f"{volatility:.2f}%")
        
        with col3:
            avg_volume = plot_data['Volume'].mean()
            st.metric("Average Volume", f"{avg_volume:,.0f}")
        
        with col4:
            price_range = ((plot_data['High'].max() - plot_data['Low'].min()) / plot_data['Close'].mean()) * 100
            st.metric("Price Range", f"{price_range:.2f}%")
    
    # Tab 4: Trading Insights
    with analysis_tabs[3]:
        st.markdown('<h3 class="subsection-header">🎯 Trading Insights & Recommendations</h3>', unsafe_allow_html=True)
        
        current_price = st.session_state.raw_data['Close'].iloc[-1]
        
        # Market sentiment analysis
        recent_returns = st.session_state.raw_data['Close'].pct_change().dropna()
        positive_days = (recent_returns > 0).sum()
        total_days = len(recent_returns)
        sentiment_score = (positive_days / total_days) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Market Sentiment Analysis:**")
            
            if sentiment_score > 60:
                sentiment = "🟢 Bullish"
                sentiment_color = "green"
            elif sentiment_score > 40:
                sentiment = "🟡 Neutral"
                sentiment_color = "orange"
            else:
                sentiment = "🔴 Bearish"
                sentiment_color = "red"
            
            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
            st.write(f"• Positive days: {positive_days}/{total_days} ({sentiment_score:.1f}%)")
            
            # Volatility assessment
            volatility = recent_returns.std() * np.sqrt(252) * 100
            if volatility > 30:
                vol_level = "🔴 High"
            elif volatility > 20:
                vol_level = "🟡 Medium"
            else:
                vol_level = "🟢 Low"
            
            st.write(f"• Volatility: {vol_level} ({volatility:.1f}%)")
            
        with col2:
            st.markdown("**🔮 Model Predictions Summary:**")
            
            if arima_available:
                arima_forecast = st.session_state.arima_results.get('forecast', {})
                if arima_forecast and arima_forecast.get('forecast'):
                    next_price_arima = arima_forecast['forecast'][0]
                    arima_direction = "📈 Up" if next_price_arima > current_price else "📉 Down"
                    st.write(f"• ARIMA Next Day: {arima_direction} ${next_price_arima:.2f}")
            
            if lstm_available:
                next_price_lstm = st.session_state.lstm_results['future_predictions'][0][0]
                lstm_direction = "📈 Up" if next_price_lstm > current_price else "📉 Down"
                st.write(f"• LSTM Next Day: {lstm_direction} ${next_price_lstm:.2f}")
        
        # Trading recommendations
        st.markdown("**💡 Trading Recommendations:**")
        
        recommendations = []
        
        # Based on sentiment
        if sentiment_score > 70:
            recommendations.append("🟢 **Buy Signal**: Strong bullish sentiment with high positive day ratio")
        elif sentiment_score < 30:
            recommendations.append("🔴 **Sell Signal**: Bearish sentiment with low positive day ratio")
        else:
            recommendations.append("🟡 **Hold**: Mixed sentiment, wait for clearer signals")
        
        # Based on volatility
        if volatility > 35:
            recommendations.append("⚠️ **High Risk**: Extreme volatility detected, consider position sizing")
        elif volatility < 15:
            recommendations.append("✅ **Low Risk**: Stable price movement, good for long-term positions")
        
        # Based on model agreement
        if arima_available and lstm_available:
            arima_forecast = st.session_state.arima_results.get('forecast', {})
            if arima_forecast and arima_forecast.get('forecast'):
                arima_next = arima_forecast['forecast'][0]
                lstm_next = st.session_state.lstm_results['future_predictions'][0][0]
                
                arima_up = arima_next > current_price
                lstm_up = lstm_next > current_price
                
                if arima_up == lstm_up:
                    direction = "upward" if arima_up else "downward"
                    recommendations.append(f"🎯 **Model Agreement**: Both models predict {direction} movement")
                else:
                    recommendations.append("🤔 **Model Disagreement**: ARIMA and LSTM predict different directions, exercise caution")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Risk metrics
        st.markdown("**⚡ Risk Metrics:**")
        
        # Value at Risk (simple 95% VaR)
        var_95 = np.percentile(recent_returns, 5) * current_price
        
        # Maximum drawdown
        cumulative_returns = (1 + recent_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value at Risk (95%)", f"${abs(var_95):.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with col3:
            sharpe_ratio = (recent_returns.mean() / recent_returns.std()) * np.sqrt(252)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Tab 5: Summary Report
    with analysis_tabs[4]:
        st.markdown('<h3 class="subsection-header">📋 Comprehensive Analysis Summary</h3>', unsafe_allow_html=True)
        
        # Generate comprehensive report
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        st.markdown(f"""
        <div class="info-card">
        <h4>📊 Analysis Report - {st.session_state.selected_stock}</h4>
        <p><strong>Generated:</strong> {report_date}</p>
        <p><strong>Analysis Period:</strong> {st.session_state.raw_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.raw_data.index[-1].strftime('%Y-%m-%d')}</p>
        <p><strong>Total Trading Days:</strong> {len(st.session_state.raw_data)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key findings
        st.markdown("**🔍 Key Findings:**")
        
        current_price = st.session_state.raw_data['Close'].iloc[-1]
        start_price = st.session_state.raw_data['Close'].iloc[0]
        total_return = ((current_price - start_price) / start_price) * 100
        
        findings = [
            f"📈 **Total Return**: {total_return:+.2f}% (${start_price:.2f} → ${current_price:.2f})",
            f"💹 **Current Price**: ${current_price:.2f}",
            f"📊 **Average Daily Volume**: {st.session_state.raw_data['Volume'].mean():,.0f}",
        ]
        
        if arima_available:
            arima_accuracy = getattr(st.session_state.arima_results.get('best_accuracy', {}), 'accuracy', 0)
            findings.append(f"📊 **ARIMA Model Accuracy**: {arima_accuracy:.2f}%")
        
        if lstm_available:
            lstm_accuracy = st.session_state.lstm_results['test_metrics']['accuracy']
            findings.append(f"🧠 **LSTM Model Accuracy**: {lstm_accuracy:.2f}%")
        
        for finding in findings:
            st.markdown(finding)
        
        # Download report option
        if st.button("📥 Generate Downloadable Report", use_container_width=True):
            
            # Create detailed report
            report_content = f"""
# Stock Analysis Report: {st.session_state.selected_stock}

**Generated:** {report_date}
**Analysis Period:** {st.session_state.raw_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.raw_data.index[-1].strftime('%Y-%m-%d')}

## Summary Statistics
- Current Price: ${current_price:.2f}
- Total Return: {total_return:+.2f}%
- Average Volume: {st.session_state.raw_data['Volume'].mean():,.0f}
- Volatility: {recent_returns.std() * np.sqrt(252) * 100:.2f}%

## Model Performance
"""
            
            if arima_available:
                arima_metrics = st.session_state.arima_results.get('best_metrics', {})
                report_content += f"""
### ARIMA Analysis
- Accuracy: {getattr(st.session_state.arima_results.get('best_accuracy', {}), 'accuracy', 0):.2f}%
- RMSE: ${arima_metrics.get('rmse', 0):.2f}
- MAE: ${arima_metrics.get('mae', 0):.2f}
- R² Score: {arima_metrics.get('r2', 0):.4f}
"""
            
            if lstm_available:
                lstm_metrics = st.session_state.lstm_results['test_metrics']
                report_content += f"""
### LSTM Analysis
- Accuracy: {lstm_metrics['accuracy']:.2f}%
- RMSE: ${lstm_metrics['rmse']:.2f}
- MAE: ${lstm_metrics['mae']:.2f}
- R² Score: {lstm_metrics['r2']:.4f}
"""
            
            # Add future predictions
            report_content += "\n## Future Predictions\n"
            
            if arima_available:
                arima_forecast = st.session_state.arima_results.get('forecast', {})
                if arima_forecast and arima_forecast.get('forecast'):
                    report_content += f"ARIMA Next Day: ${arima_forecast['forecast'][0]:.2f}\n"
            
            if lstm_available:
                lstm_next = st.session_state.lstm_results['future_predictions'][0][0]
                report_content += f"LSTM Next Day: ${lstm_next:.2f}\n"
            
            # Provide download
            st.download_button(
                label="📄 Download Report as Text",
                data=report_content,
                file_name=f"{st.session_state.selected_stock}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# Interactive Analysis Button
if st.session_state.data_loaded:
    if st.button("🎯 Launch Interactive Analysis", use_container_width=True):
        interactive_analysis_comprehensive()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 <strong>Advanced Stock Analysis Platform</strong></p>
    <p>Built with Streamlit • Yahoo Finance API • TensorFlow/Keras • Statsmodels • Plotly</p>
    <p><em>For educational and research purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tips:**")
    st.sidebar.markdown("• Start by selecting a stock and date range")
    st.sidebar.markdown("• Load data and explore with preprocessing")
    st.sidebar.markdown("• Run ARIMA analysis for statistical insights")
    st.sidebar.markdown("• Use LSTM for neural network predictions")
    st.sidebar.markdown("• Compare models in interactive analysis")
    
    # Quick start button
    if st.sidebar.button("🚀 Quick Start Guide", use_container_width=True):
        st.info("""
        **🚀 Quick Start Guide:**
        
        1. **📊 Select Stock**: Choose from NVDA, MSFT, or META
        2. **📅 Set Date Range**: Select analysis period (default: 1 year)
        3. **🔄 Load Data**: Click 'Load Stock Data' button
        4. **📈 Explore Data**: Use the 4 tabs to understand your data
        5. **🛠️ Preprocess**: Add technical indicators and transformations
        6. **📊 ARIMA Analysis**: Run statistical time series analysis
        7. **🧠 LSTM Analysis**: Train neural network for predictions
        8. **🎯 Interactive Analysis**: Compare models and get insights
        
        **💡 Pro Tips:**
        - ARIMA works best with stationary data (use differencing)
        - LSTM can handle non-stationary data automatically
        - Compare both models for comprehensive insights
        - Use interactive analysis for trading recommendations
        """)
    
    # Display current session state
    if st.sidebar.checkbox("🔍 Show Session State", help="Debug information"):
        st.sidebar.markdown("**Session State:**")
        st.sidebar.write(f"Data Loaded: {'✅' if st.session_state.data_loaded else '❌'}")
        st.sidebar.write(f"ARIMA Results: {'✅' if hasattr(st.session_state, 'arima_results') and st.session_state.arima_results else '❌'}")
        st.sidebar.write(f"LSTM Results: {'✅' if hasattr(st.session_state, 'lstm_results') and st.session_state.lstm_results else '❌'}")
        
        if st.session_state.data_loaded:
            st.sidebar.write(f"Stock: {st.session_state.selected_stock}")
            st.sidebar.write(f"Data Points: {len(st.session_state.raw_data)}")
            st.sidebar.write(f"Date Range: {st.session_state.raw_data.index[0].strftime('%Y-%m-%d')} to {st.session_state.raw_data.index[-1].strftime('%Y-%m-%d')}")