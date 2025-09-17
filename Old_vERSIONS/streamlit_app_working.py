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

# For modeling - without TensorFlow initially
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Statsmodels for ARIMA
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
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=min(5, len(analysis_df))).mean()
    analysis_df['Price_MA'] = analysis_df['Close'].rolling(window=min(5, len(analysis_df))).mean()
    
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
        avg_return = analysis_df['Daily_Return'].mean() * 100 if len(analysis_df) > 1 else 0
        st.metric("📈 Avg Daily Return", f"{avg_return:.3f}%")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Analysis", "📊 Volume Analysis", "🎯 Technical Analysis", "📉 Risk Analysis", "🔍 Pattern Analysis"])
    
    with tab1:
        st.subheader("📈 Price Movement Analysis")
        
        # Price charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily returns bar chart
            if len(analysis_df) > 1:
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
        
        if len(analysis_df) > 2:
            fig_candle.add_trace(go.Scatter(
                x=analysis_df.index,
                y=analysis_df['Price_MA'],
                mode='lines',
                name='Moving Average',
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
            # Volume pie chart by day of week
            if len(analysis_df) > 1:
                analysis_df['DayOfWeek'] = analysis_df.index.day_name()
                vol_by_day = analysis_df.groupby('DayOfWeek')['Volume'].sum()
                fig_vol_pie = px.pie(
                    values=vol_by_day.values,
                    names=vol_by_day.index,
                    title="Volume Distribution by Day of Week"
                )
                fig_vol_pie.update_layout(height=400)
                st.plotly_chart(fig_vol_pie, use_container_width=True)
        
        # Price vs Volume correlation
        if len(analysis_df) > 1:
            fig_corr = px.scatter(
                analysis_df,
                x='Volume',
                y='Close',
                size='Price_Range',
                title="Price vs Volume Correlation",
                labels={'Volume': 'Trading Volume', 'Close': 'Closing Price ($)'},
                trendline="ols"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Technical Analysis")
        
        # Simple moving averages
        if len(analysis_df) > 5:
            analysis_df['SMA_3'] = analysis_df['Close'].rolling(window=3).mean()
            analysis_df['SMA_5'] = analysis_df['Close'].rolling(window=5).mean()
            
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['Close'],
                mode='lines', name='Close Price', line=dict(color='blue')
            ))
            fig_sma.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['SMA_3'],
                mode='lines', name='3-Day SMA', line=dict(color='red')
            ))
            fig_sma.add_trace(go.Scatter(
                x=analysis_df.index, y=analysis_df['SMA_5'],
                mode='lines', name='5-Day SMA', line=dict(color='green')
            ))
            
            fig_sma.update_layout(
                title="Simple Moving Averages",
                yaxis_title="Price ($)",
                height=500
            )
            st.plotly_chart(fig_sma, use_container_width=True)
        
        # Price momentum
        if len(analysis_df) > 1:
            analysis_df['Momentum'] = analysis_df['Close'] - analysis_df['Close'].shift(1)
            fig_momentum = px.bar(
                analysis_df.dropna(),
                x=analysis_df.dropna().index,
                y='Momentum',
                title="Price Momentum (Daily Change)",
                color='Momentum',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
    
    with tab4:
        st.subheader("📉 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            if len(analysis_df) > 1 and 'Daily_Return' in analysis_df.columns:
                fig_dist = px.histogram(
                    analysis_df.dropna(),
                    x='Daily_Return',
                    nbins=min(10, len(analysis_df)//2),
                    title="Daily Returns Distribution",
                    labels={'Daily_Return': 'Daily Return', 'count': 'Frequency'}
                )
                if len(analysis_df.dropna()) > 0:
                    fig_dist.add_vline(x=analysis_df['Daily_Return'].mean(), 
                                     line_dash="dash", line_color="red", 
                                     annotation_text="Mean")
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Risk metrics
            returns = analysis_df['Daily_Return'].dropna()
            if len(returns) > 0:
                st.write("📊 **Risk Metrics**")
                st.write(f"• **Volatility (Daily)**: {returns.std() * 100:.2f}%")
                if returns.std() > 0:
                    st.write(f"• **Sharpe Ratio**: {returns.mean() / returns.std():.2f}")
                st.write(f"• **Max Daily Gain**: {returns.max() * 100:.2f}%")
                st.write(f"• **Max Daily Loss**: {returns.min() * 100:.2f}%")
                st.write(f"• **Positive Days**: {(returns > 0).sum()} / {len(returns)}")
                if len(returns) >= 5:
                    st.write(f"• **VaR (95%)**: {np.percentile(returns, 5) * 100:.2f}%")
            
            # Risk level visualization
            if len(returns) > 0:
                volatility = returns.std() * 100
                if volatility < 1:
                    risk_level = "Low"
                    risk_color = "green"
                elif volatility < 3:
                    risk_level = "Medium"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    risk_color = "red"
                
                st.markdown(f"**Risk Level**: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    with tab5:
        st.subheader("🔍 Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Candlestick pattern analysis
            if len(analysis_df) > 0:
                patterns = {
                    'Small Body': (analysis_df['Body_Size'] < (analysis_df['Price_Range'] * 0.3)).sum(),
                    'Large Body': (analysis_df['Body_Size'] > (analysis_df['Price_Range'] * 0.7)).sum(),
                    'Upper Shadow Dominant': (analysis_df['Upper_Shadow'] > analysis_df['Lower_Shadow']).sum(),
                    'Lower Shadow Dominant': (analysis_df['Lower_Shadow'] > analysis_df['Upper_Shadow']).sum(),
                }
                
                fig_patterns = px.pie(
                    values=list(patterns.values()),
                    names=list(patterns.keys()),
                    title="Candlestick Patterns Distribution"
                )
                st.plotly_chart(fig_patterns, use_container_width=True)
        
        with col2:
            # Trading signals
            st.write("🚦 **Trading Signals**")
            if len(analysis_df) > 1:
                latest_return = analysis_df['Daily_Return'].iloc[-1] if not pd.isna(analysis_df['Daily_Return'].iloc[-1]) else 0
                volume_trend = "High" if analysis_df['Volume'].iloc[-1] > analysis_df['Volume'].mean() else "Low"
                
                if latest_return > 0.02:
                    st.success("📈 Strong Bullish Signal")
                elif latest_return > 0:
                    st.info("📈 Mild Bullish Signal")
                elif latest_return < -0.02:
                    st.error("📉 Strong Bearish Signal")
                elif latest_return < 0:
                    st.warning("📉 Mild Bearish Signal")
                else:
                    st.info("⚪ Neutral Signal")
                
                st.write(f"• **Volume Trend**: {volume_trend}")
                st.write(f"• **Price Trend**: {'Up' if price_change > 0 else 'Down'}")
        
        # Volume-Price relationship
        if len(analysis_df) > 1:
            price_volume_corr = analysis_df['Close'].corr(analysis_df['Volume'])
            st.write(f"**📊 Price-Volume Correlation**: {price_volume_corr:.3f}")
            
            if abs(price_volume_corr) > 0.5:
                st.success("💪 Strong correlation between price and volume")
            elif abs(price_volume_corr) > 0.3:
                st.info("📊 Moderate correlation between price and volume")
            else:
                st.warning("📉 Weak correlation between price and volume")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return analysis_df

# Simple ML models for prediction (without TensorFlow)
def build_ml_models(data):
    """Build simple ML models for prediction"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 Machine Learning Predictions</h2>', unsafe_allow_html=True)
    
    try:
        # Prepare data
        df = data.copy()
        df['Target'] = df['Close'].shift(-1)  # Next day's price
        df = df.dropna()
        
        if len(df) < 10:
            st.warning("⚠️ Not enough data for ML predictions. Need at least 10 data points.")
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
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predict
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'mae': mae
            }
        
        # Display results
        col1, col2 = st.columns(2)
        
        for i, (name, result) in enumerate(results.items()):
            with col1 if i == 0 else col2:
                st.subheader(f"🎯 {name}")
                st.metric("MSE", f"{result['mse']:.2f}")
                st.metric("MAE", f"${result['mae']:.2f}")
        
        # Plot predictions
        fig = go.Figure()
        
        # Test data dates
        test_dates = df.index[train_size:]
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test,
            mode='lines',
            name='Actual Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        colors = ['red', 'orange']
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=result['predictions'],
                mode='lines',
                name=f'{name} Predictions',
                line=dict(color=colors[i], dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="🎯 ML Model Predictions vs Actual Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Predictions (Next 5 days)")
        
        # Get last data point for prediction
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler_X.transform(last_features)
        
        st.write("📅 **ML Model Predictions:**")
        for name, result in results.items():
            future_pred_scaled = result['model'].predict(last_features_scaled)
            future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1))[0, 0]
            
            change = future_pred - data['Close'].iloc[-1]
            change_pct = (change / data['Close'].iloc[-1]) * 100
            direction = "📈" if change > 0 else "📉"
            
            st.write(f"{direction} **{name}**: **${future_pred:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return results
        
    except Exception as e:
        st.error(f"❌ Error in ML models: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return None

# ARIMA Model (same as before)
def build_arima_model(data):
    """Build and train ARIMA model"""
    if not STATSMODELS_AVAILABLE:
        st.error("❌ Statsmodels not available. Cannot build ARIMA model.")
        return None
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 ARIMA Statistical Model</h2>', unsafe_allow_html=True)
    
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
        
        # Split data
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Simple ARIMA parameters
        best_params = (1, diff_order, 1)
        st.write(f"**Using ARIMA parameters**: {best_params}")
        
        # Fit ARIMA model
        with st.spinner("🎯 Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=best_params)
            fitted_arima = arima_model.fit()
        
        # Make predictions
        forecast_steps = len(test_data)
        forecast = fitted_arima.forecast(steps=forecast_steps)
        
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
            mode='lines', name='ARIMA Forecast', line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="📊 ARIMA Model Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("🔮 ARIMA Future Prediction")
        future_forecast = fitted_arima.forecast(steps=1)[0]
        change = future_forecast - data['Close'].iloc[-1]
        change_pct = (change / data['Close'].iloc[-1]) * 100
        direction = "📈" if change > 0 else "📉"
        
        st.write(f"{direction} **ARIMA Prediction**: **${future_forecast:.2f}** (Change: ${change:.2f}, {change_pct:+.2f}%)")
        
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
            st.markdown('<h2 class="section-header">🔮 Prediction Models</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("🤖 Select Prediction Model", ["Machine Learning", "ARIMA Statistical", "Both Models"])
            with col2:
                st.write("**Model Information:**")
                if model_choice == "Machine Learning":
                    st.info("🧠 Linear Regression & Random Forest models")
                elif model_choice == "ARIMA Statistical":
                    st.info("📊 Statistical model for time series forecasting")
                else:
                    st.info("🔄 Compare both model types")
            
            if st.button("🚀 Train & Predict", use_container_width=True):
                if model_choice in ["Machine Learning", "Both Models"]:
                    ml_results = build_ml_models(processed_data)
                
                if model_choice in ["ARIMA Statistical", "Both Models"]:
                    arima_model = build_arima_model(processed_data)

if __name__ == "__main__":
    main()
