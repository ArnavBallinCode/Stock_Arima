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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats

# LSTM imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropouty
    LSTM_AVAILABLE = True
    # Configure TensorFlow for better compatibility
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except ImportError:
    LSTM_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis & Prediction",
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
    .analysis-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Advanced Stock Analysis & Prediction Platform</h1>', unsafe_allow_html=True)

if not LSTM_AVAILABLE:
    st.warning("⚠️ TensorFlow not available. LSTM predictions will be disabled. Only ARIMA predictions will work.")

# Sidebar for stock selection and parameters
st.sidebar.header("Configuration")

# Stock selection
stock_options = {
    "NVIDIA": "NVDA",
    "OpenAI (Microsoft)": "MSFT",
    "X (Twitter) - Meta": "META"
}

selected_stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
stock_symbol = stock_options[selected_stock_name]

# Date range selection
st.sidebar.subheader("Date Range")
end_date = st.sidebar.date_input("End Date", datetime.now())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

# Analysis period selection
st.sidebar.subheader("Analysis Period")
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

# Enhanced data preprocessing function
def preprocess_data(df):
    """Comprehensive data preprocessing with detailed analysis"""
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Original data info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
        st.write(f"**Trading Days:** {len(df)} days")
        
    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)
        
    with col3:
        st.subheader("Memory Usage")
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.write(f"**Total Memory:** {memory_usage:.2f} MB")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        st.warning("Missing values found!")
        
        # Create missing values visualization
        fig_missing = px.bar(
            x=missing_values.index,
            y=missing_values.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig_missing, use_container_width=True)
        
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
            
        st.success("Missing values filled!")
    else:
        st.success("✅ No missing values found!")
    
    # Data quality checks
    st.subheader("Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Check for negative prices
        negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
        if negative_prices:
            st.error("❌ Negative prices found!")
        else:
            st.success("✅ All prices are positive")
        
        # Check for zero volume
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume > 0:
            st.warning(f"⚠️ {zero_volume} days with zero volume")
        else:
            st.success("✅ No zero volume days")
    
    with col2:
        # Price logic validation
        price_logic = ((df['High'] >= df['Low']) & 
                       (df['High'] >= df['Open']) & 
                       (df['High'] >= df['Close']) &
                       (df['Low'] <= df['Open']) & 
                       (df['Low'] <= df['Close'])).all()
        
        if price_logic:
            st.success("✅ Price relationships are logical")
        else:
            st.error("❌ Some price relationships are illogical!")
        
        # Remove duplicates
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = initial_shape - df.shape[0]
        
        if removed_duplicates > 0:
            st.info(f"ℹ️ Removed {removed_duplicates} duplicate rows")
        else:
            st.success("✅ No duplicate rows found")
    
    return df

# Enhanced analysis function with comprehensive charts
def analyze_data_comprehensive(df, analysis_type, analysis_date=None, analysis_start=None, analysis_end=None):
    """Comprehensive data analysis with multiple chart types"""
    st.markdown('<h2 class="section-header">📊 Comprehensive Data Analysis</h2>', unsafe_allow_html=True)
    
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
    
    # Calculate additional metrics
    analysis_df['Daily_Return'] = analysis_df['Close'].pct_change()
    analysis_df['Price_Range'] = analysis_df['High'] - analysis_df['Low']
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=5).mean()
    analysis_df['Volatility'] = analysis_df['Daily_Return'].rolling(window=5).std()
    
    # Summary statistics
    st.subheader(f"📈 Analysis Summary - {period_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    price_change = analysis_df['Close'].iloc[-1] - analysis_df['Close'].iloc[0] if len(analysis_df) > 1 else 0
    price_change_pct = (price_change / analysis_df['Close'].iloc[0]) * 100 if len(analysis_df) > 1 else 0
    avg_volume = analysis_df['Volume'].mean()
    max_price = analysis_df['High'].max()
    min_price = analysis_df['Low'].min()
    avg_volatility = analysis_df['Volatility'].mean() if 'Volatility' in analysis_df.columns else 0
    
    with col1:
        st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    with col2:
        st.metric("Average Volume", f"{avg_volume:,.0f}")
    with col3:
        st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
    with col4:
        st.metric("Average Volatility", f"{avg_volatility:.4f}")
    with col5:
        st.metric("Trading Days", f"{len(analysis_df)}")
    
    # 1. Price Distribution Analysis
    st.subheader("💰 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution histogram
        fig_hist = px.histogram(
            analysis_df, x='Close',
            title="Closing Price Distribution",
            nbins=20,
            labels={'Close': 'Closing Price ($)', 'count': 'Frequency'}
        )
        fig_hist.update_traces(opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Price statistics box plot
        price_data = [analysis_df['Open'], analysis_df['High'], analysis_df['Low'], analysis_df['Close']]
        fig_box = go.Figure()
        
        for i, (prices, name) in enumerate(zip(price_data, ['Open', 'High', 'Low', 'Close'])):
            fig_box.add_trace(go.Box(y=prices, name=name, boxpoints='outliers'))
        
        fig_box.update_layout(title="Price Components Box Plot", yaxis_title="Price ($)")
        st.plotly_chart(fig_box, use_container_width=True)
    
    # 2. Volume Analysis
    st.subheader("📊 Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume distribution
        fig_vol_hist = px.histogram(
            analysis_df, x='Volume',
            title="Volume Distribution",
            nbins=20,
            labels={'Volume': 'Trading Volume', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_vol_hist, use_container_width=True)
    
    with col2:
        # Volume trend over time
        fig_vol_trend = px.line(
            analysis_df, x=analysis_df.index, y='Volume',
            title="Volume Trend Over Time",
            labels={'x': 'Date', 'Volume': 'Trading Volume'}
        )
        st.plotly_chart(fig_vol_trend, use_container_width=True)
    
    # 3. Returns Analysis
    if len(analysis_df) > 1:
        st.subheader("📈 Returns Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns histogram
            fig_returns = px.histogram(
                analysis_df.dropna(), x='Daily_Return',
                title="Daily Returns Distribution",
                nbins=20,
                labels={'Daily_Return': 'Daily Return (%)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Returns vs Volume scatter
            fig_scatter = px.scatter(
                analysis_df.dropna(), x='Volume', y='Daily_Return',
                title="Returns vs Volume Relationship",
                labels={'Volume': 'Trading Volume', 'Daily_Return': 'Daily Return (%)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 4. Technical Indicators Pie Chart
    st.subheader("🥧 Technical Analysis Summary")
    
    # Calculate technical indicators
    positive_days = (analysis_df['Daily_Return'] > 0).sum()
    negative_days = (analysis_df['Daily_Return'] < 0).sum()
    flat_days = (analysis_df['Daily_Return'] == 0).sum()
    
    # High volume days (above 80th percentile)
    volume_threshold = analysis_df['Volume'].quantile(0.8)
    high_volume_days = (analysis_df['Volume'] > volume_threshold).sum()
    normal_volume_days = len(analysis_df) - high_volume_days
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price movement pie chart
        if positive_days + negative_days + flat_days > 0:
            fig_pie_returns = px.pie(
                values=[positive_days, negative_days, flat_days],
                names=['Positive Days', 'Negative Days', 'Flat Days'],
                title="Price Movement Distribution",
                color_discrete_sequence=['#00CC96', '#EF553B', '#FFA15A']
            )
            st.plotly_chart(fig_pie_returns, use_container_width=True)
    
    with col2:
        # Volume activity pie chart
        fig_pie_volume = px.pie(
            values=[high_volume_days, normal_volume_days],
            names=['High Volume', 'Normal Volume'],
            title="Volume Activity Distribution",
            color_discrete_sequence=['#AB63FA', '#19D3F3']
        )
        st.plotly_chart(fig_pie_volume, use_container_width=True)
    
    # 5. Advanced Metrics Bar Chart
    st.subheader("📊 Advanced Metrics Comparison")
    
    # Calculate metrics for bar chart
    metrics_data = {
        'Metric': ['Average Price', 'Volatility (×100)', 'Volume (×1000)', 'Price Range', 'Max Daily Return (×100)'],
        'Value': [
            analysis_df['Close'].mean(),
            analysis_df['Volatility'].mean() * 100 if 'Volatility' in analysis_df.columns else 0,
            analysis_df['Volume'].mean() / 1000,
            analysis_df['Price_Range'].mean(),
            abs(analysis_df['Daily_Return']).max() * 100 if 'Daily_Return' in analysis_df.columns else 0
        ]
    }
    
    fig_metrics = px.bar(
        x=metrics_data['Metric'],
        y=metrics_data['Value'],
        title="Key Metrics Overview",
        labels={'x': 'Metrics', 'y': 'Normalized Values'}
    )
    fig_metrics.update_traces(marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # 6. Correlation Heatmap
    st.subheader("🔥 Correlation Analysis")
    
    # Calculate correlations
    corr_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Daily_Return' in analysis_df.columns:
        corr_columns.append('Daily_Return')
    
    corr_matrix = analysis_df[corr_columns].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_heatmap.update_layout(width=600, height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 7. Statistical Summary Table
    st.subheader("📋 Statistical Summary")
    
    summary_stats = analysis_df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
    st.dataframe(summary_stats, use_container_width=True)
    
    return analysis_df

# LSTM Model with enhanced error handling
def build_lstm_model(data, look_back=60):
    """Build and train LSTM model with comprehensive analysis"""
    if not LSTM_AVAILABLE:
        st.error("❌ TensorFlow is not available. Cannot build LSTM model.")
        return None, None
    
    st.markdown('<h2 class="section-header">🤖 LSTM Neural Network Prediction</h2>', unsafe_allow_html=True)
    
    if len(data) < look_back + 20:
        st.warning(f"⚠️ Need at least {look_back + 20} data points for LSTM. Current: {len(data)}")
        return None, None
    
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
        
        if len(X) < 10:
            st.warning("Not enough sequences for training. Need more data.")
            return None, None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 50
                progress_bar.progress(progress)
                status_text.text(f'Training progress: {epoch + 1}/50 epochs')
        
        with st.spinner("Training LSTM model..."):
            history = model.fit(
                X_train, y_train, 
                batch_size=32, 
                epochs=50, 
                verbose=0,
                validation_data=(X_test, y_test),
                callbacks=[ProgressCallback()]
            )
        
        progress_bar.empty()
        status_text.empty()
        
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
            st.metric("Train MSE", f"{train_mse:.2f}")
        with col2:
            st.metric("Test MSE", f"{test_mse:.2f}")
        with col3:
            st.metric("Train MAE", f"{train_mae:.2f}")
        with col4:
            st.metric("Test MAE", f"{test_mae:.2f}")
        
        # Plot training history
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = px.line(
                x=range(len(history.history['loss'])),
                y=history.history['loss'],
                title="Training Loss",
                labels={'x': 'Epoch', 'y': 'Loss'}
            )
            if 'val_loss' in history.history:
                fig_loss.add_scatter(
                    x=list(range(len(history.history['val_loss']))),
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss'
                )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Plot predictions
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=data.index[look_back:look_back+len(y_train_actual)],
            y=y_train_actual.flatten(),
            mode='lines',
            name='Actual (Train)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index[look_back+len(y_train_actual):look_back+len(y_train_actual)+len(y_test_actual)],
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual (Test)',
            line=dict(color='blue')
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=data.index[look_back:look_back+len(train_predict)],
            y=train_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Train)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)],
            y=test_predict.flatten(),
            mode='lines',
            name='LSTM Prediction (Test)',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.update_layout(
            title="LSTM Model Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Price Predictions (Next 5 days)")
        
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
        
        # Display future predictions in a nice format
        future_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Price': [f"${price[0]:.2f}" for price in future_predictions],
            'Change from Last': [f"${(future_predictions[i][0] - data['Close'].iloc[-1]):.2f}" for i in range(5)]
        })
        
        st.dataframe(future_df, use_container_width=True)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"Error building LSTM model: {e}")
        return None, None

# Enhanced ARIMA model
def build_arima_model(data):
    """Build and train ARIMA model with comprehensive analysis"""
    st.markdown('<h2 class="section-header">📊 ARIMA Statistical Model Prediction</h2>', unsafe_allow_html=True)
    
    # Prepare data
    prices = data['Close']
    
    # Stationarity test
    st.subheader("📈 Stationarity Analysis")
    result = adfuller(prices.dropna())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ADF Statistic:** {result[0]:.6f}")
        st.write(f"**p-value:** {result[1]:.6f}")
        st.write(f"**Critical Values:**")
        for key, value in result[4].items():
            st.write(f"  - {key}: {value:.3f}")
    
    with col2:
        if result[1] <= 0.05:
            st.success("✅ Series is stationary")
            diff_order = 0
        else:
            st.warning("⚠️ Series is not stationary. Applying differencing...")
            diff_order = 1
            prices_diff = prices.diff().dropna()
            result_diff = adfuller(prices_diff)
            st.write(f"**After differencing:**")
            st.write(f"ADF Statistic: {result_diff[0]:.6f}")
            st.write(f"p-value: {result_diff[1]:.6f}")
    
    # Split data
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    # Parameter selection
    st.subheader("🔧 Model Configuration")
    auto_arima = st.checkbox("Use Auto ARIMA (recommended)", value=True)
    
    if auto_arima:
        # Grid search for best parameters
        best_aic = float('inf')
        best_params = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_combinations = 3 * 2 * 3  # p, d, q ranges
        current_combination = 0
        
        with st.spinner("Finding optimal ARIMA parameters..."):
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            current_combination += 1
                            progress = current_combination / total_combinations
                            progress_bar.progress(progress)
                            status_text.text(f'Testing ARIMA({p},{d},{q})... {current_combination}/{total_combinations}')
                            
                            model = ARIMA(train_data, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
        
        progress_bar.empty()
        status_text.empty()
        
        if best_params:
            st.success(f"✅ Optimal parameters found: ARIMA{best_params} (AIC: {best_aic:.2f})")
        else:
            best_params = (1, 1, 1)
            st.warning("⚠️ Could not find optimal parameters. Using default (1,1,1)")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("AR order (p)", 0, 5, 1)
        with col2:
            d = st.slider("Differencing order (d)", 0, 2, 1)
        with col3:
            q = st.slider("MA order (q)", 0, 5, 1)
        best_params = (p, d, q)
    
    # Fit ARIMA model
    try:
        with st.spinner("Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=best_params)
            fitted_arima = arima_model.fit()
        
        st.success("✅ ARIMA model trained successfully!")
        
        # Model diagnostics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Summary")
            st.text(str(fitted_arima.summary().tables[1]))
        
        with col2:
            st.subheader("📈 Model Information")
            st.write(f"**AIC:** {fitted_arima.aic:.2f}")
            st.write(f"**BIC:** {fitted_arima.bic:.2f}")
            st.write(f"**Log Likelihood:** {fitted_arima.llf:.2f}")
        
        # Make predictions
        forecast_steps = len(test_data)
        forecast = fitted_arima.forecast(steps=forecast_steps)
        forecast_ci = fitted_arima.get_forecast(steps=forecast_steps).conf_int()
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecast)
        mae = mean_absolute_error(test_data, forecast)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test MSE", f"{mse:.2f}")
        with col2:
            st.metric("Test MAE", f"{mae:.2f}")
        
        # Plot results
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data.values,
            mode='lines',
            name='Training Data',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Actual Test Data',
            line=dict(color='green')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=forecast,
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='red', dash='dash')
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
            title="ARIMA Model Predictions with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Price Predictions (Next 5 days)")
        future_forecast = fitted_arima.forecast(steps=5)
        future_ci = fitted_arima.get_forecast(steps=5).conf_int()
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
        
        # Display future predictions in a nice format
        future_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Price': [f"${price:.2f}" for price in future_forecast],
            'Lower CI (95%)': [f"${ci_lower:.2f}" for ci_lower in future_ci.iloc[:, 0]],
            'Upper CI (95%)': [f"${ci_upper:.2f}" for ci_upper in future_ci.iloc[:, 1]],
            'Change from Last': [f"${(price - data['Close'].iloc[-1]):.2f}" for price in future_forecast]
        })
        
        st.dataframe(future_df, use_container_width=True)
        
        return fitted_arima
        
    except Exception as e:
        st.error(f"❌ Error fitting ARIMA model: {e}")
        return None

# Main application
def main():
    # Load data
    if st.sidebar.button("📥 Load Data", type="primary"):
        with st.spinner("Loading stock data..."):
            data = load_stock_data(stock_symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            st.session_state['data'] = data
            st.session_state['stock_name'] = selected_stock_name
            st.balloons()
            st.success(f"✅ Data loaded successfully for {selected_stock_name} ({stock_symbol})")
            
            # Quick data overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 **Records:** {len(data)}")
            with col2:
                st.info(f"💰 **Latest Price:** ${data['Close'].iloc[-1]:.2f}")
            with col3:
                change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                change_pct = (change / data['Close'].iloc[0]) * 100
                st.info(f"📈 **Total Change:** {change_pct:.2f}%")
        else:
            st.error("❌ Failed to load data. Please check your internet connection and try again.")
    
    # Process data if loaded
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Display raw data
        st.markdown('<h2 class="section-header">📋 Raw Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Data preview with statistics
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(data, use_container_width=True)
        
        with col2:
            st.markdown(
                f"""
                <div class="analysis-box">
                <h4>📊 Dataset Info</h4>
                <p><strong>Rows:</strong> {data.shape[0]}</p>
                <p><strong>Columns:</strong> {data.shape[1]}</p>
                <p><strong>Start Date:</strong> {data.index.min().date()}</p>
                <p><strong>End Date:</strong> {data.index.max().date()}</p>
                <p><strong>Price Range:</strong> ${data['Low'].min():.2f} - ${data['High'].max():.2f}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Data preprocessing
        if st.button("🔧 Preprocess Data", type="primary"):
            processed_data = preprocess_data(data.copy())
            st.session_state['processed_data'] = processed_data
            st.balloons()
        
        # Analysis section
        if 'processed_data' in st.session_state:
            processed_data = st.session_state['processed_data']
            
            if st.button("📊 Run Comprehensive Analysis", type="primary"):
                if analysis_type == "Specific Day":
                    analysis_df = analyze_data_comprehensive(processed_data, analysis_type, analysis_date)
                elif analysis_type == "Specific Week":
                    analysis_df = analyze_data_comprehensive(processed_data, analysis_type, analysis_date)
                else:
                    analysis_df = analyze_data_comprehensive(processed_data, analysis_type, 
                                                           analysis_start=analysis_start, 
                                                           analysis_end=analysis_end)
                
                if analysis_df is not None and not analysis_df.empty:
                    st.session_state['analysis_df'] = analysis_df
                    st.balloons()
            
            # Model prediction section
            st.markdown('<h2 class="section-header">🔮 Prediction Models</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    <div class="analysis-box">
                    <h4>🤖 LSTM Neural Network</h4>
                    <p>• Deep learning approach</p>
                    <p>• Captures complex patterns</p>
                    <p>• Best for non-linear trends</p>
                    <p>• Requires more data</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    """
                    <div class="analysis-box">
                    <h4>📊 ARIMA Statistical Model</h4>
                    <p>• Traditional econometric method</p>
                    <p>• Statistical significance</p>
                    <p>• Confidence intervals</p>
                    <p>• Works with less data</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            model_choice = st.selectbox("🎯 Select Prediction Model", ["ARIMA Only", "LSTM Only", "Both Models"])
            
            if st.button("🚀 Train & Predict", type="primary"):
                if model_choice in ["LSTM Only", "Both Models"] and LSTM_AVAILABLE:
                    lstm_model, scaler = build_lstm_model(processed_data)
                elif model_choice in ["LSTM Only", "Both Models"] and not LSTM_AVAILABLE:
                    st.error("❌ LSTM not available. TensorFlow is required.")
                
                if model_choice in ["ARIMA Only", "Both Models"]:
                    arima_model = build_arima_model(processed_data)
                
                st.balloons()
                st.success("🎉 Model training completed!")

if __name__ == "__main__":
    main()
