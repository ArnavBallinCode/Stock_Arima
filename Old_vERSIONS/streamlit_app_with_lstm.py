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

# For modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Safe TensorFlow import with configuration to avoid mutex issues
TENSORFLOW_AVAILABLE = False
try:
    # Configure environment before TensorFlow import
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    import tensorflow as tf
    
    # Single-threaded configuration to prevent mutex locks
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Disable GPU to avoid GPU-related mutex issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices([], 'GPU')
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    TENSORFLOW_AVAILABLE = True
    
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow not available: {e}")

# Statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Complete Stock Analysis with LSTM",
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
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .accuracy-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .model-comparison {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Complete Stock Analysis with LSTM & Accuracy Metrics</h1>', unsafe_allow_html=True)

# Display available models in sidebar
st.sidebar.markdown("### 🤖 Available Models")
if TENSORFLOW_AVAILABLE:
    st.sidebar.success("✅ LSTM Neural Network")
else:
    st.sidebar.error("❌ LSTM Neural Network")

if STATSMODELS_AVAILABLE:
    st.sidebar.success("✅ ARIMA Statistical")
else:
    st.sidebar.error("❌ ARIMA Statistical")

st.sidebar.success("✅ Linear Regression")
st.sidebar.success("✅ Random Forest")

# Stock selection
st.sidebar.header("🔧 Configuration")

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

# Calculate comprehensive accuracy metrics
def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics for model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # Direction accuracy
    if len(y_true) > 1:
        actual_direction = np.sign(y_true[1:] - y_true[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    # Overall accuracy (R² transformed to percentage)
    overall_accuracy = max(0, r2 * 100)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Direction Accuracy': direction_accuracy,
        'Overall Accuracy': overall_accuracy
    }

# Data preprocessing function
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Shape**: {df.shape}")
        st.write(f"**Date Range**: {df.index.min().date()} to {df.index.max().date()}")
    with col2:
        st.write(f"**Missing Values**: {df.isnull().sum().sum()}")
        st.write(f"**Duplicates**: {df.duplicated().sum()}")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')
        st.success("✅ Missing values filled")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Data validation
    if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
        st.warning("⚠️ Negative prices detected")
    else:
        st.success("✅ Data validation passed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return df

# Enhanced analysis with charts
def enhanced_analysis(df, analysis_type, analysis_date=None, analysis_start=None, analysis_end=None):
    """Enhanced analysis with multiple visualizations"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 Enhanced Data Analysis</h2>', unsafe_allow_html=True)
    
    # Filter data
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
        return None
    
    st.subheader(f"📈 Analysis for {period_name}")
    
    # Calculate metrics
    analysis_df = analysis_df.copy()
    analysis_df['Daily_Return'] = analysis_df['Close'].pct_change()
    analysis_df['Price_Range'] = analysis_df['High'] - analysis_df['Low']
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=min(5, len(analysis_df))).mean()
    
    # Key metrics
    price_change = analysis_df['Close'].iloc[-1] - analysis_df['Close'].iloc[0]
    price_change_pct = (price_change / analysis_df['Close'].iloc[0]) * 100
    volatility = analysis_df['Close'].std()
    volume_avg = analysis_df['Volume'].mean()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("📊 Volatility", f"${volatility:.2f}")
    with col3:
        st.metric("📦 Avg Volume", f"{volume_avg:,.0f}")
    with col4:
        avg_return = analysis_df['Daily_Return'].mean() * 100 if len(analysis_df) > 1 else 0
        st.metric("📈 Avg Return", f"{avg_return:.3f}%")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Price & Volume", "📊 Analysis Charts", "🔍 Technical Indicators"])
    
    with tab1:
        # Candlestick with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Stock Price', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        fig.add_trace(go.Candlestick(
            x=analysis_df.index,
            open=analysis_df['Open'],
            high=analysis_df['High'],
            low=analysis_df['Low'],
            close=analysis_df['Close'],
            name="Price"
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=analysis_df.index,
            y=analysis_df['Volume'],
            name="Volume",
            marker_color='rgba(158,202,225,0.8)'
        ), row=2, col=1)
        
        fig.update_layout(height=600, title=f"Price & Volume - {period_name}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns
            if len(analysis_df) > 1:
                fig_returns = px.bar(
                    analysis_df.dropna(),
                    x=analysis_df.dropna().index,
                    y='Daily_Return',
                    title="Daily Returns",
                    color='Daily_Return',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Price range
            fig_range = px.bar(
                analysis_df,
                x=analysis_df.index,
                y='Price_Range',
                title="Daily Price Range",
                color='Price_Range',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_range, use_container_width=True)
        
        # Returns distribution
        if len(analysis_df) > 5:
            fig_dist = px.histogram(
                analysis_df.dropna(),
                x='Daily_Return',
                title="Returns Distribution",
                nbins=min(10, len(analysis_df)//2)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        # Technical indicators
        if len(analysis_df) > 5:
            analysis_df['SMA_5'] = analysis_df['Close'].rolling(window=5).mean()
            
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['Close'],
                mode='lines', name='Close Price', line=dict(color='blue')
            ))
            fig_tech.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['SMA_5'],
                mode='lines', name='5-Day SMA', line=dict(color='red')
            ))
            
            fig_tech.update_layout(title="Price with Moving Average", height=400)
            st.plotly_chart(fig_tech, use_container_width=True)
        
        # Risk metrics
        if len(analysis_df) > 1:
            returns = analysis_df['Daily_Return'].dropna()
            if len(returns) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Daily Volatility", f"{returns.std() * 100:.2f}%")
                    st.metric("Max Gain", f"{returns.max() * 100:.2f}%")
                with col2:
                    st.metric("Max Loss", f"{returns.min() * 100:.2f}%")
                    positive_pct = (returns > 0).mean() * 100
                    st.metric("Positive Days", f"{positive_pct:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return analysis_df

# LSTM Model with proper error handling
def build_lstm_model(data, look_back=60):
    """Build LSTM model with comprehensive accuracy metrics"""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available. LSTM model cannot be built.")
        return None, None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🧠 LSTM Neural Network with Accuracy Metrics</h2>', unsafe_allow_html=True)
    
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
            st.warning("⚠️ Insufficient data for LSTM. Need at least 80 data points.")
            st.markdown('</div>', unsafe_allow_html=True)
            return None, None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with explicit CPU usage
        with tf.device('/CPU:0'):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        with st.spinner("🎯 Training LSTM model..."):
            with tf.device('/CPU:0'):
                history = model.fit(
                    X_train, y_train,
                    batch_size=16,
                    epochs=20,
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
        
        # Calculate accuracy metrics
        train_metrics = calculate_accuracy_metrics(y_train_actual.flatten(), train_predict.flatten())
        test_metrics = calculate_accuracy_metrics(y_test_actual.flatten(), test_predict.flatten())
        
        # Display accuracy metrics
        st.markdown('<div class="accuracy-box">', unsafe_allow_html=True)
        st.subheader("🎯 LSTM Accuracy Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Accuracy:**")
            st.write(f"• **Overall Accuracy**: {train_metrics['Overall Accuracy']:.2f}%")
            st.write(f"• **R² Score**: {train_metrics['R²']:.4f}")
            st.write(f"• **Direction Accuracy**: {train_metrics['Direction Accuracy']:.2f}%")
            st.write(f"• **MAPE**: {train_metrics['MAPE']:.2f}%")
            st.write(f"• **RMSE**: ${train_metrics['RMSE']:.2f}")
        
        with col2:
            st.write("**Testing Accuracy:**")
            st.write(f"• **Overall Accuracy**: {test_metrics['Overall Accuracy']:.2f}%")
            st.write(f"• **R² Score**: {test_metrics['R²']:.4f}")
            st.write(f"• **Direction Accuracy**: {test_metrics['Direction Accuracy']:.2f}%")
            st.write(f"• **MAPE**: {test_metrics['MAPE']:.2f}%")
            st.write(f"• **RMSE**: ${test_metrics['RMSE']:.2f}")
        
        # Overall performance assessment
        overall_acc = test_metrics['Overall Accuracy']
        if overall_acc > 80:
            st.success(f"🎯 **Excellent LSTM Performance**: {overall_acc:.1f}% accuracy!")
        elif overall_acc > 60:
            st.info(f"🎯 **Good LSTM Performance**: {overall_acc:.1f}% accuracy")
        elif overall_acc > 40:
            st.warning(f"🎯 **Moderate LSTM Performance**: {overall_acc:.1f}% accuracy")
        else:
            st.error(f"🎯 **Poor LSTM Performance**: {overall_acc:.1f}% accuracy")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot predictions
        fig = go.Figure()
        
        test_dates = data.index[look_back + train_size:]
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict.flatten(),
            mode='lines',
            name=f'LSTM Prediction (Acc: {overall_acc:.1f}%)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f"🧠 LSTM Predictions vs Actual (Accuracy: {overall_acc:.1f}%)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 LSTM Future Prediction")
        last_sequence = scaled_data[-look_back:]
        with tf.device('/CPU:0'):
            future_pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        future_price = scaler.inverse_transform(future_pred)[0, 0]
        
        current_price = data['Close'].iloc[-1]
        change = future_price - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.success(f"{direction} **LSTM Next Day Prediction** (Acc: {overall_acc:.1f}%): **${future_price:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model, scaler, test_metrics
        
    except Exception as e:
        st.error(f"❌ Error in LSTM model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None, None

# Machine Learning models with accuracy
def build_ml_models(data):
    """Build ML models with comprehensive accuracy metrics"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models with Accuracy</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        df = data.copy()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        if len(df) < 10:
            st.warning("⚠️ Insufficient data for ML models.")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
        
        # Features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features].values
        y = df['Target'].values
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = calculate_accuracy_metrics(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }
        
        # Display results
        st.markdown('<div class="accuracy-box">', unsafe_allow_html=True)
        st.subheader("🎯 ML Model Accuracy Comparison")
        
        for name, result in results.items():
            with st.expander(f"📊 {name} - Accuracy: {result['metrics']['Overall Accuracy']:.1f}%"):
                metrics = result['metrics']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Accuracy Metrics:**")
                    st.write(f"• **Overall Accuracy**: {metrics['Overall Accuracy']:.2f}%")
                    st.write(f"• **R² Score**: {metrics['R²']:.4f}")
                    st.write(f"• **Direction Accuracy**: {metrics['Direction Accuracy']:.2f}%")
                
                with col2:
                    st.write("**Error Metrics:**")
                    st.write(f"• **MAPE**: {metrics['MAPE']:.2f}%")
                    st.write(f"• **RMSE**: ${metrics['RMSE']:.2f}")
                    st.write(f"• **MAE**: ${metrics['MAE']:.2f}")
                
                # Performance assessment
                acc = metrics['Overall Accuracy']
                if acc > 80:
                    st.success(f"🎯 Excellent {name} performance!")
                elif acc > 60:
                    st.info(f"🎯 Good {name} performance")
                else:
                    st.warning(f"🎯 {name} needs improvement")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison chart
        fig = go.Figure()
        
        test_dates = df.index[train_size:]
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=3)
        ))
        
        colors = ['red', 'orange']
        for i, (name, result) in enumerate(results.items()):
            acc = result['metrics']['Overall Accuracy']
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=result['predictions'],
                mode='lines',
                name=f'{name} (Acc: {acc:.1f}%)',
                line=dict(color=colors[i], dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="🤖 ML Model Predictions Comparison",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 ML Future Predictions")
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        current_price = data['Close'].iloc[-1]
        
        for name, result in results.items():
            future_pred = result['model'].predict(last_features_scaled)[0]
            change = future_pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉"
            acc = result['metrics']['Overall Accuracy']
            
            st.write(f"{direction} **{name}** (Acc: {acc:.1f}%): **${future_pred:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return results
        
    except Exception as e:
        st.error(f"❌ Error in ML models: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None

# ARIMA model with accuracy
def build_arima_model(data):
    """Build ARIMA model with accuracy metrics"""
    if not STATSMODELS_AVAILABLE:
        st.error("❌ Statsmodels not available for ARIMA.")
        return None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 ARIMA Model with Accuracy</h2>', unsafe_allow_html=True)
    
    try:
        prices = data['Close']
        
        # Split data
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Fit ARIMA
        with st.spinner("🎯 Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=(1, 1, 1))
            fitted_arima = arima_model.fit()
        
        # Predictions
        forecast = fitted_arima.forecast(steps=len(test_data))
        
        # Calculate accuracy
        metrics = calculate_accuracy_metrics(test_data.values, forecast)
        
        # Display accuracy
        st.markdown('<div class="accuracy-box">', unsafe_allow_html=True)
        st.subheader("🎯 ARIMA Accuracy Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Accuracy Metrics:**")
            st.write(f"• **Overall Accuracy**: {metrics['Overall Accuracy']:.2f}%")
            st.write(f"• **R² Score**: {metrics['R²']:.4f}")
            st.write(f"• **Direction Accuracy**: {metrics['Direction Accuracy']:.2f}%")
        
        with col2:
            st.write("**Error Metrics:**")
            st.write(f"• **MAPE**: {metrics['MAPE']:.2f}%")
            st.write(f"• **RMSE**: ${metrics['RMSE']:.2f}")
            st.write(f"• **MAE**: ${metrics['MAE']:.2f}")
        
        # Performance assessment
        acc = metrics['Overall Accuracy']
        if acc > 80:
            st.success(f"🎯 Excellent ARIMA performance: {acc:.1f}%!")
        elif acc > 60:
            st.info(f"🎯 Good ARIMA performance: {acc:.1f}%")
        else:
            st.warning(f"🎯 ARIMA performance: {acc:.1f}% - needs improvement")
        
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
            mode='lines', name=f'ARIMA Forecast (Acc: {acc:.1f}%)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f"📊 ARIMA Predictions (Accuracy: {acc:.1f}%)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 ARIMA Future Prediction")
        future_forecast = fitted_arima.forecast(steps=1)[0]
        current_price = data['Close'].iloc[-1]
        change = future_forecast - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.success(f"{direction} **ARIMA Next Day Prediction** (Acc: {acc:.1f}%): **${future_forecast:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return fitted_arima, metrics
        
    except Exception as e:
        st.error(f"❌ Error in ARIMA model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None

# Main application
def main():
    # Load data
    if st.sidebar.button("🔄 Load Data", use_container_width=True):
        with st.spinner("📥 Loading stock data..."):
            data = load_stock_data(stock_symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            st.session_state['data'] = data
            st.session_state['stock_name'] = selected_stock_name
            st.success(f"✅ Data loaded for **{selected_stock_name}** ({stock_symbol})")
            
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Records", len(data))
            with col2:
                st.metric("📅 Days", (data.index[-1] - data.index[0]).days)
            with col3:
                st.metric("💰 Latest Price", f"${data['Close'].iloc[-1]:.2f}")
        else:
            st.error("❌ Failed to load data")
    
    # Process data
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Dataset overview
        st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
        with st.expander("📊 View Dataset", expanded=False):
            st.dataframe(data, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Basic Statistics:**")
                st.dataframe(data.describe())
            with col2:
                st.write("**Data Quality:**")
                st.write(f"• Missing values: {data.isnull().sum().sum()}")
                st.write(f"• Duplicate rows: {data.duplicated().sum()}")
                st.write(f"• Date range: {(data.index[-1] - data.index[0]).days} days")
        
        # Preprocessing
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
                
                if analysis_df is not None:
                    st.session_state['analysis_df'] = analysis_df
            
            # Model predictions with accuracy
            st.markdown('<h2 class="section-header">🔮 ML Predictions with Accuracy Metrics</h2>', unsafe_allow_html=True)
            
            # Model selection
            available_models = ["Machine Learning Models", "ARIMA Statistical"]
            if TENSORFLOW_AVAILABLE:
                available_models.extend(["LSTM Neural Network", "All Models"])
            
            model_choice = st.selectbox("🤖 Select Model Type", available_models)
            
            if st.button("🚀 Train Models & Show Accuracy", use_container_width=True):
                model_results = {}
                
                # Train selected models
                if model_choice in ["Machine Learning Models", "All Models"]:
                    ml_results = build_ml_models(processed_data)
                    if ml_results:
                        for name, result in ml_results.items():
                            model_results[name] = result['metrics']['Overall Accuracy']
                
                if model_choice in ["ARIMA Statistical", "All Models"]:
                    arima_model, arima_metrics = build_arima_model(processed_data)
                    if arima_metrics:
                        model_results["ARIMA"] = arima_metrics['Overall Accuracy']
                
                if model_choice in ["LSTM Neural Network", "All Models"] and TENSORFLOW_AVAILABLE:
                    lstm_model, scaler, lstm_metrics = build_lstm_model(processed_data)
                    if lstm_metrics:
                        model_results["LSTM"] = lstm_metrics['Overall Accuracy']
                
                # Model comparison
                if model_results:
                    st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
                    st.markdown('<h3 class="section-header">🏆 Final Model Accuracy Ranking</h3>', unsafe_allow_html=True)
                    
                    # Sort by accuracy
                    sorted_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display ranking
                    for i, (model, accuracy) in enumerate(sorted_models):
                        if i == 0:
                            st.success(f"🥇 **Champion**: {model} - **{accuracy:.1f}% accuracy**")
                        elif i == 1:
                            st.info(f"🥈 **Runner-up**: {model} - **{accuracy:.1f}% accuracy**")
                        elif i == 2:
                            st.warning(f"🥉 **Third Place**: {model} - **{accuracy:.1f}% accuracy**")
                        else:
                            st.write(f"📊 {model} - {accuracy:.1f}% accuracy")
                    
                    # Accuracy comparison chart
                    fig_comparison = px.bar(
                        x=list(model_results.values()),
                        y=list(model_results.keys()),
                        orientation='h',
                        title="🎯 Model Accuracy Comparison",
                        labels={'x': 'Accuracy (%)', 'y': 'Model'},
                        color=list(model_results.values()),
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 100]
                    )
                    fig_comparison.update_layout(height=400)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Best model recommendation
                    best_model, best_accuracy = sorted_models[0]
                    if best_accuracy > 80:
                        st.success(f"🎯 **Recommendation**: Use {best_model} for predictions (Excellent {best_accuracy:.1f}% accuracy)")
                    elif best_accuracy > 60:
                        st.info(f"🎯 **Recommendation**: Use {best_model} for predictions (Good {best_accuracy:.1f}% accuracy)")
                    else:
                        st.warning(f"🎯 **Caution**: Best model {best_model} has only {best_accuracy:.1f}% accuracy. Consider more data or different approaches.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
