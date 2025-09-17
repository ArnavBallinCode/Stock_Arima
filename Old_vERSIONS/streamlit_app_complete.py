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

# For modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# TensorFlow import with proper error handling and configuration
TENSORFLOW_AVAILABLE = False
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    
    import tensorflow as tf
    # Configure TensorFlow for stable operation
    tf.config.set_visible_devices([], 'GPU')  # Force CPU usage to avoid GPU issues
    tf.get_logger().setLevel('ERROR')  # Suppress warnings
    
    # Set threading configuration
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    TENSORFLOW_AVAILABLE = True
    st.sidebar.success("✅ LSTM available")
except Exception as e:
    st.sidebar.warning(f"⚠️ LSTM unavailable: {str(e)[:50]}...")

# Statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
    st.sidebar.success("✅ ARIMA available")
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.sidebar.warning("⚠️ ARIMA unavailable")

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
    .model-comparison {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Complete Stock Analysis & Prediction Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Data • Advanced Analytics • LSTM/ARIMA/ML Predictions • Professional Charts</p>', unsafe_allow_html=True)

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

# Calculate accuracy metrics
def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Accuracy as percentage (100 - MAPE)
    accuracy = max(0, 100 - mape)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Accuracy': accuracy
    }

# Data preprocessing function (same as before but more compact)
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Check and handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"Found {missing_values.sum()} missing values. Filling with forward fill method.")
        df = df.fillna(method='ffill').fillna(method='bfill')
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
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if not negative_prices:
        st.success("✅ All prices are positive and data is clean!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return df

# Enhanced analysis function (simplified for space)
def enhanced_analysis(df, analysis_type, analysis_date=None, analysis_start=None, analysis_end=None):
    """Enhanced analysis with comprehensive visualizations"""
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
        return None
    
    st.subheader(f"📈 Analysis for {period_name}")
    
    # Calculate metrics
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
        st.metric("📈 Avg Return", f"{avg_return:.3f}%")
    
    # Create comprehensive charts
    tab1, tab2, tab3 = st.tabs(["📈 Price & Volume", "📊 Technical Analysis", "📉 Risk Analysis"])
    
    with tab1:
        # Candlestick with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Movement', 'Trading Volume'),
            row_heights=[0.7, 0.3]
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
        
        fig.update_layout(height=600, title_text="Price and Volume Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(analysis_df) > 5:
            # Moving averages
            analysis_df['MA5'] = analysis_df['Close'].rolling(window=5).mean()
            analysis_df['MA10'] = analysis_df['Close'].rolling(window=min(10, len(analysis_df))).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df['Close'], name='Close', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df['MA5'], name='5-Day MA', line=dict(color='red')))
            if len(analysis_df) > 10:
                fig.add_trace(go.Scatter(x=analysis_df.index, y=analysis_df['MA10'], name='10-Day MA', line=dict(color='green')))
            
            fig.update_layout(title="Moving Averages", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(analysis_df) > 1:
            # Returns distribution
            fig = px.histogram(analysis_df.dropna(), x='Daily_Return', 
                             title="Daily Returns Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return analysis_df

# LSTM Model with proper configuration
def build_lstm_model(data, look_back=60):
    """Build LSTM model with comprehensive accuracy metrics"""
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow not available. Cannot build LSTM model.")
        return None, None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 LSTM Neural Network Model</h2>', unsafe_allow_html=True)
    
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
            st.warning(f"⚠️ Not enough data for LSTM. Need at least {look_back + 20} data points, got {len(data)}.")
            return None, None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with proper configuration
        with tf.device('/CPU:0'):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        progress_bar = st.progress(0)
        with st.spinner("🎯 Training LSTM model..."):
            history = model.fit(
                X_train, y_train, 
                batch_size=16, 
                epochs=30, 
                verbose=0,
                validation_split=0.1,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
            )
            progress_bar.progress(100)
        
        # Make predictions
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        
        # Transform back to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate comprehensive metrics
        train_metrics = calculate_accuracy_metrics(y_train_actual, train_predict.flatten())
        test_metrics = calculate_accuracy_metrics(y_test_actual, test_predict.flatten())
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Training Performance")
            st.metric("Accuracy", f"{train_metrics['Accuracy']:.1f}%")
            st.metric("R² Score", f"{train_metrics['R²']:.3f}")
            st.metric("RMSE", f"${train_metrics['RMSE']:.2f}")
            st.metric("MAE", f"${train_metrics['MAE']:.2f}")
        
        with col2:
            st.subheader("🔍 Testing Performance")
            st.metric("Accuracy", f"{test_metrics['Accuracy']:.1f}%")
            st.metric("R² Score", f"{test_metrics['R²']:.3f}")
            st.metric("RMSE", f"${test_metrics['RMSE']:.2f}")
            st.metric("MAE", f"${test_metrics['MAE']:.2f}")
        
        # Plot predictions
        fig = go.Figure()
        
        # Test data indices
        test_dates = data.index[look_back + train_size:look_back + train_size + len(test_predict)]
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test_actual,
            mode='lines', name='Actual Prices', line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_predict.flatten(),
            mode='lines', name='LSTM Predictions', line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f"🤖 LSTM Model Performance (Accuracy: {test_metrics['Accuracy']:.1f}%)",
            xaxis_title="Date", yaxis_title="Price ($)", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 LSTM Future Prediction")
        last_sequence = scaled_data[-look_back:]
        future_pred_scaled = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        future_pred = scaler.inverse_transform(future_pred_scaled)[0, 0]
        
        change = future_pred - data['Close'].iloc[-1]
        change_pct = (change / data['Close'].iloc[-1]) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.write(f"{direction} **Next Day Prediction**: **${future_pred:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model, scaler, test_metrics
        
    except Exception as e:
        st.error(f"❌ Error in LSTM model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None, None

# Enhanced ML models with accuracy metrics
def build_ml_models(data):
    """Build ML models with comprehensive accuracy metrics"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎯 Machine Learning Models</h2>', unsafe_allow_html=True)
    
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
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
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
        
        # Display results in comparison format
        st.subheader("📊 Model Comparison")
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['metrics']['Accuracy']:.1f}%",
                'R² Score': f"{result['metrics']['R²']:.3f}",
                'RMSE': f"${result['metrics']['RMSE']:.2f}",
                'MAE': f"${result['metrics']['MAE']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Plot comparison
        fig = go.Figure()
        test_dates = df.index[train_size:]
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test,
            mode='lines', name='Actual Prices', line=dict(color='blue', width=2)
        ))
        
        colors = ['red', 'orange']
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=test_dates, y=result['predictions'],
                mode='lines', name=f'{name} (Acc: {result["metrics"]["Accuracy"]:.1f}%)',
                line=dict(color=colors[i], dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="🎯 ML Models Performance Comparison",
            xaxis_title="Date", yaxis_title="Price ($)", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return results
        
    except Exception as e:
        st.error(f"❌ Error in ML models: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None

# ARIMA model with accuracy metrics
def build_arima_model(data):
    """Build ARIMA model with comprehensive accuracy metrics"""
    if not STATSMODELS_AVAILABLE:
        st.error("❌ Statsmodels not available. Cannot build ARIMA model.")
        return None, None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 ARIMA Statistical Model</h2>', unsafe_allow_html=True)
    
    try:
        prices = data['Close']
        
        # Check stationarity
        result = adfuller(prices.dropna())
        diff_order = 0 if result[1] <= 0.05 else 1
        
        # Split data
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Fit ARIMA model
        best_params = (1, diff_order, 1)
        
        with st.spinner("🎯 Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=best_params)
            fitted_arima = arima_model.fit()
        
        # Make predictions
        forecast = fitted_arima.forecast(steps=len(test_data))
        
        # Calculate metrics
        metrics = calculate_accuracy_metrics(test_data.values, forecast)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.1f}%")
        with col2:
            st.metric("📊 R² Score", f"{metrics['R²']:.3f}")
        with col3:
            st.metric("📈 RMSE", f"${metrics['RMSE']:.2f}")
        with col4:
            st.metric("📉 MAE", f"${metrics['MAE']:.2f}")
        
        # Plot results
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values,
            mode='lines', name='Actual Prices', line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index, y=forecast,
            mode='lines', name=f'ARIMA Forecast (Acc: {metrics["Accuracy"]:.1f}%)',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f"📊 ARIMA Model Performance (Accuracy: {metrics['Accuracy']:.1f}%)",
            xaxis_title="Date", yaxis_title="Price ($)", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return fitted_arima, metrics
        
    except Exception as e:
        st.error(f"❌ Error in ARIMA model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None

# Model comparison function
def compare_all_models(lstm_metrics, ml_results, arima_metrics):
    """Compare all models side by side"""
    st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🏆 Complete Model Comparison</h2>', unsafe_allow_html=True)
    
    comparison_data = []
    
    # Add LSTM if available
    if lstm_metrics:
        comparison_data.append({
            'Model': 'LSTM Neural Network',
            'Type': 'Deep Learning',
            'Accuracy': f"{lstm_metrics['Accuracy']:.1f}%",
            'R² Score': f"{lstm_metrics['R²']:.3f}",
            'RMSE': f"${lstm_metrics['RMSE']:.2f}",
            'MAE': f"${lstm_metrics['MAE']:.2f}"
        })
    
    # Add ML models if available
    if ml_results:
        for name, result in ml_results.items():
            model_type = 'Ensemble' if 'Forest' in name else 'Linear'
            comparison_data.append({
                'Model': name,
                'Type': model_type,
                'Accuracy': f"{result['metrics']['Accuracy']:.1f}%",
                'R² Score': f"{result['metrics']['R²']:.3f}",
                'RMSE': f"${result['metrics']['RMSE']:.2f}",
                'MAE': f"${result['metrics']['MAE']:.2f}"
            })
    
    # Add ARIMA if available
    if arima_metrics:
        comparison_data.append({
            'Model': 'ARIMA',
            'Type': 'Statistical',
            'Accuracy': f"{arima_metrics['Accuracy']:.1f}%",
            'R² Score': f"{arima_metrics['R²']:.3f}",
            'RMSE': f"${arima_metrics['RMSE']:.2f}",
            'MAE': f"${arima_metrics['MAE']:.2f}"
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        st.subheader("📊 Performance Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Find best model
        accuracy_values = [float(row['Accuracy'].replace('%', '')) for row in comparison_data]
        best_idx = accuracy_values.index(max(accuracy_values))
        best_model = comparison_data[best_idx]['Model']
        best_accuracy = comparison_data[best_idx]['Accuracy']
        
        st.success(f"🏆 **Best Performing Model**: {best_model} with {best_accuracy} accuracy!")
        
        # Create accuracy bar chart
        fig = px.bar(
            comparison_df, x='Model', y=[float(acc.replace('%', '')) for acc in comparison_df['Accuracy']],
            title="Model Accuracy Comparison", color='Type',
            labels={'y': 'Accuracy (%)'}, text=[acc for acc in comparison_df['Accuracy']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Records", len(data))
            with col2:
                st.metric("📅 Days", f"{(data.index[-1] - data.index[0]).days}")
            with col3:
                st.metric("💰 Latest Price", f"${data['Close'].iloc[-1]:.2f}")
            with col4:
                price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("📈 Total Change", f"{price_change:+.2f}%")
        else:
            st.error("❌ Failed to load data. Please check your internet connection and try again.")
    
    # Process data if loaded
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Display raw data overview
        st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
        
        with st.expander("📊 View Dataset", expanded=False):
            tab1, tab2 = st.tabs(["📈 Data Preview", "📊 Statistics"])
            with tab1:
                st.dataframe(data, use_container_width=True)
            with tab2:
                st.dataframe(data.describe(), use_container_width=True)
        
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
                
                if analysis_df is not None:
                    st.session_state['analysis_df'] = analysis_df
            
            # Model prediction section
            st.markdown('<h2 class="section-header">🤖 AI/ML Prediction Models</h2>', unsafe_allow_html=True)
            
            # Model selection
            available_models = []
            if TENSORFLOW_AVAILABLE:
                available_models.append("LSTM Neural Network")
            available_models.extend(["Machine Learning Models", "ARIMA Statistical"])
            available_models.append("All Available Models")
            
            model_choice = st.selectbox("🤖 Select Models to Train", available_models)
            
            if st.button("🚀 Train & Compare Models", use_container_width=True):
                lstm_metrics = None
                ml_results = None
                arima_metrics = None
                
                if model_choice in ["LSTM Neural Network", "All Available Models"] and TENSORFLOW_AVAILABLE:
                    _, _, lstm_metrics = build_lstm_model(processed_data)
                
                if model_choice in ["Machine Learning Models", "All Available Models"]:
                    ml_results = build_ml_models(processed_data)
                
                if model_choice in ["ARIMA Statistical", "All Available Models"] and STATSMODELS_AVAILABLE:
                    _, arima_metrics = build_arima_model(processed_data)
                
                # Compare all models if multiple were trained
                if model_choice == "All Available Models" or (lstm_metrics and ml_results) or (lstm_metrics and arima_metrics) or (ml_results and arima_metrics):
                    compare_all_models(lstm_metrics, ml_results, arima_metrics)

if __name__ == "__main__":
    main()
