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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set page configuration
st.set_page_config(
    page_title="Stock Data Analysis & Prediction",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Stock Data Analysis & Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar for stock selection and parameters
st.sidebar.header("Configuration")

# Stock selection
stock_options = {
    "NVIDIA": "NVDA",
    "OpenAI (Microsoft)": "MSFT", # OpenAI is private, using Microsoft as proxy
    "X (Twitter) - Meta": "META"  # X is private, using Meta as similar social media proxy
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

# Data preprocessing function
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Original data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Data Info")
        st.write(f"Shape: {df.shape}")
        st.write(f"Date Range: {df.index.min()} to {df.index.max()}")
        
    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)
    
    # Check for missing values
    st.subheader("Missing Values Analysis")
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
            
        st.success("Missing values filled!")
    else:
        st.success("No missing values found!")
    
    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    removed_duplicates = initial_shape - df.shape[0]
    
    if removed_duplicates > 0:
        st.info(f"Removed {removed_duplicates} duplicate rows")
    else:
        st.success("No duplicate rows found!")
    
    # Data validation
    st.subheader("Data Validation")
    
    # Check for negative prices
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()
    if negative_prices:
        st.warning("Warning: Negative prices found in data!")
    else:
        st.success("All prices are positive")
    
    # Check for logical price relationships
    price_logic = ((df['High'] >= df['Low']) & 
                   (df['High'] >= df['Open']) & 
                   (df['High'] >= df['Close']) &
                   (df['Low'] <= df['Open']) & 
                   (df['Low'] <= df['Close'])).all()
    
    if price_logic:
        st.success("Price relationships are logical (High >= Open, Close, Low)")
    else:
        st.warning("Warning: Some price relationships seem illogical!")
    
    return df

# Analysis function
def analyze_data(df, analysis_type, analysis_date=None, analysis_start=None, analysis_end=None):
    """Analyze data for spikes and trends"""
    st.markdown('<h2 class="section-header">📊 Data Analysis</h2>', unsafe_allow_html=True)
    
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
    
    st.subheader(f"Analysis for {period_name}")
    
    # Calculate metrics
    price_change = analysis_df['Close'].iloc[-1] - analysis_df['Close'].iloc[0]
    price_change_pct = (price_change / analysis_df['Close'].iloc[0]) * 100
    volatility = analysis_df['Close'].std()
    volume_avg = analysis_df['Volume'].mean()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    with col2:
        st.metric("Volatility", f"${volatility:.2f}")
    
    with col3:
        st.metric("Average Volume", f"{volume_avg:,.0f}")
    
    with col4:
        max_price = analysis_df['High'].max()
        min_price = analysis_df['Low'].min()
        st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
    
    # Identify spikes and significant movements
    st.subheader("Significant Movements & Spikes")
    
    # Calculate daily returns
    analysis_df['Daily_Return'] = analysis_df['Close'].pct_change()
    analysis_df['Price_Spike'] = analysis_df['High'] - analysis_df['Low']
    analysis_df['Volume_Spike'] = analysis_df['Volume'] > analysis_df['Volume'].quantile(0.8)
    
    # Find significant price movements (>2% change)
    significant_moves = analysis_df[abs(analysis_df['Daily_Return']) > 0.02]
    
    if not significant_moves.empty:
        st.write("Days with significant price movements (>2%):")
        for idx, row in significant_moves.iterrows():
            direction = "📈" if row['Daily_Return'] > 0 else "📉"
            st.write(f"{direction} {idx.date()}: {row['Daily_Return']*100:.2f}% change")
    else:
        st.info("No significant price movements found in this period")
    
    # Volume spikes
    volume_spikes = analysis_df[analysis_df['Volume_Spike']]
    if not volume_spikes.empty:
        st.write("Days with high volume activity:")
        for idx, row in volume_spikes.iterrows():
            st.write(f"📊 {idx.date()}: Volume {row['Volume']:,.0f}")
    
    return analysis_df

# Visualization function
def create_visualizations(df, analysis_df, period_name):
    """Create various charts and visualizations"""
    st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    # 1. Price Chart with Volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Stock Price', 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=analysis_df.index,
            open=analysis_df['Open'],
            high=analysis_df['High'],
            low=analysis_df['Low'],
            close=analysis_df['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=analysis_df.index,
            y=analysis_df['Volume'],
            name="Volume",
            marker_color='rgba(158,202,225,0.8)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Stock Analysis - {period_name}",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        xaxis_title="Date",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Moving Averages
    if len(analysis_df) > 5:
        analysis_df['MA5'] = analysis_df['Close'].rolling(window=5).mean()
        analysis_df['MA10'] = analysis_df['Close'].rolling(window=min(10, len(analysis_df))).mean()
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=analysis_df.index,
            y=analysis_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))
        
        fig2.add_trace(go.Scatter(
            x=analysis_df.index,
            y=analysis_df['MA5'],
            mode='lines',
            name='5-Day MA',
            line=dict(color='red', dash='dash')
        ))
        
        if len(analysis_df) > 10:
            fig2.add_trace(go.Scatter(
                x=analysis_df.index,
                y=analysis_df['MA10'],
                mode='lines',
                name='10-Day MA',
                line=dict(color='green', dash='dot')
            ))
        
        fig2.update_layout(
            title="Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Daily Returns Distribution
    if 'Daily_Return' in analysis_df.columns:
        fig3 = px.histogram(
            analysis_df.dropna(),
            x='Daily_Return',
            title="Daily Returns Distribution",
            labels={'Daily_Return': 'Daily Return (%)', 'count': 'Frequency'}
        )
        fig3.update_traces(opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Price vs Volume Scatter
    fig4 = px.scatter(
        analysis_df,
        x='Volume',
        y='Close',
        title="Price vs Volume Relationship",
        labels={'Volume': 'Trading Volume', 'Close': 'Closing Price ($)'}
    )
    st.plotly_chart(fig4, use_container_width=True)

# LSTM Model
def build_lstm_model(data, look_back=60):
    """Build and train LSTM model"""
    st.markdown('<h2 class="section-header">🤖 LSTM Model Prediction</h2>', unsafe_allow_html=True)
    
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
        st.warning("Not enough data for LSTM training. Need at least 70 data points.")
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
    
    # Train model
    with st.spinner("Training LSTM model..."):
        history = model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
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
        title="LSTM Model Predictions",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Future predictions
    st.subheader("Future Predictions (Next 5 days)")
    
    # Get last sequence
    last_sequence = scaled_data[-look_back:]
    future_predictions = []
    
    for _ in range(5):
        next_pred = model.predict(last_sequence.reshape(1, look_back, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
    
    # Display future predictions
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        st.write(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price[0]:.2f}")
    
    return model, scaler

# ARIMA Model
def build_arima_model(data):
    """Build and train ARIMA model"""
    st.markdown('<h2 class="section-header">📊 ARIMA Model Prediction</h2>', unsafe_allow_html=True)
    
    # Prepare data
    prices = data['Close']
    
    # Check stationarity
    st.subheader("Stationarity Test")
    result = adfuller(prices.dropna())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"ADF Statistic: {result[0]:.6f}")
        st.write(f"p-value: {result[1]:.6f}")
    
    with col2:
        if result[1] <= 0.05:
            st.success("Series is stationary")
            diff_order = 0
        else:
            st.warning("Series is not stationary. Applying differencing...")
            diff_order = 1
    
    # If not stationary, difference the series
    if diff_order > 0:
        prices_diff = prices.diff().dropna()
        result_diff = adfuller(prices_diff)
        st.write(f"After differencing - ADF Statistic: {result_diff[0]:.6f}, p-value: {result_diff[1]:.6f}")
    
    # Split data
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    # Auto ARIMA or manual parameter selection
    st.subheader("ARIMA Parameters")
    auto_arima = st.checkbox("Use Auto ARIMA (recommended)", value=True)
    
    if auto_arima:
        # Simple parameter search
        best_aic = float('inf')
        best_params = None
        
        with st.spinner("Finding best ARIMA parameters..."):
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
            st.success(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        else:
            best_params = (1, 1, 1)
            st.warning("Could not find optimal parameters. Using default (1,1,1)")
    else:
        p = st.slider("AR order (p)", 0, 5, 1)
        d = st.slider("Differencing order (d)", 0, 2, 1)
        q = st.slider("MA order (q)", 0, 5, 1)
        best_params = (p, d, q)
    
    # Fit ARIMA model
    try:
        with st.spinner("Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=best_params)
            fitted_arima = arima_model.fit()
        
        # Model summary
        st.subheader("Model Summary")
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
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title="ARIMA Model Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("Future Predictions (Next 5 days)")
        future_forecast = fitted_arima.forecast(steps=5)
        future_ci = fitted_arima.get_forecast(steps=5).conf_int()
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')
        
        for i, (date, price, ci_lower, ci_upper) in enumerate(zip(future_dates, future_forecast, 
                                                                  future_ci.iloc[:, 0], future_ci.iloc[:, 1])):
            st.write(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f} (CI: ${ci_lower:.2f} - ${ci_upper:.2f})")
        
        return fitted_arima
        
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {e}")
        return None

# Main application
def main():
    # Load data
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading stock data..."):
            data = load_stock_data(stock_symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            st.session_state['data'] = data
            st.session_state['stock_name'] = selected_stock_name
            st.success(f"Data loaded successfully for {selected_stock_name} ({stock_symbol})")
        else:
            st.error("Failed to load data. Please check your internet connection and try again.")
    
    # Process data if loaded
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Display raw data
        st.markdown('<h2 class="section-header">📋 Raw Dataset</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(data, use_container_width=True)
        
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write(f"Period: {data.index.min().date()} to {data.index.max().date()}")
        
        # Data preprocessing
        if st.button("Preprocess Data"):
            processed_data = preprocess_data(data.copy())
            st.session_state['processed_data'] = processed_data
        
        # Analysis
        if 'processed_data' in st.session_state:
            processed_data = st.session_state['processed_data']
            
            if st.button("Analyze Data"):
                if analysis_type == "Specific Day":
                    analysis_df = analyze_data(processed_data, analysis_type, analysis_date)
                elif analysis_type == "Specific Week":
                    analysis_df = analyze_data(processed_data, analysis_type, analysis_date)
                else:
                    analysis_df = analyze_data(processed_data, analysis_type, 
                                             analysis_start=analysis_start, analysis_end=analysis_end)
                
                if analysis_df is not None and not analysis_df.empty:
                    st.session_state['analysis_df'] = analysis_df
                    
                    # Create visualizations
                    if analysis_type == "Specific Day":
                        period_name = f"Day: {analysis_date}"
                    elif analysis_type == "Specific Week":
                        period_name = f"Week: {analysis_date}"
                    else:
                        period_name = f"Period: {analysis_start} to {analysis_end}"
                    
                    create_visualizations(processed_data, analysis_df, period_name)
            
            # Model prediction section
            st.markdown('<h2 class="section-header">🔮 Prediction Models</h2>', unsafe_allow_html=True)
            
            model_choice = st.selectbox("Select Prediction Model", ["LSTM", "ARIMA", "Both"])
            
            if st.button("Train & Predict"):
                if model_choice in ["LSTM", "Both"]:
                    lstm_model, scaler = build_lstm_model(processed_data)
                
                if model_choice in ["ARIMA", "Both"]:
                    arima_model = build_arima_model(processed_data)

if __name__ == "__main__":
    main()
