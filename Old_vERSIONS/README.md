# 📈 Stock Data Analysis & Prediction Platform

A comprehensive Streamlit web application for stock data collection, preprocessing, analysis, and prediction using Yahoo Finance API with LSTM and ARIMA models.

## 🌟 Features

### 📊 Data Collection
- **Three Stock Options**: NVIDIA (NVDA), OpenAI proxy (MSFT), X proxy (META)
- **Yahoo Finance Integration**: Real-time and historical stock data
- **Flexible Date Ranges**: Custom date selection for data collection

### 🔧 Data Preprocessing
- **Missing Value Detection**: Comprehensive null value analysis
- **Data Cleaning**: Multiple fill strategies (Forward Fill, Backward Fill, Interpolate, Mean)
- **Duplicate Removal**: Automatic duplicate detection and removal
- **Data Validation**: Price logic validation and negative value detection
- **Type Conversion**: Automatic data type optimization

### 📈 Analysis & Visualization
- **Time Period Analysis**: 
  - Specific day analysis
  - Weekly analysis  
  - Custom date range analysis
- **Spike Detection**: Identifies significant price movements (>2% changes)
- **Volume Analysis**: High volume activity detection
- **Interactive Charts**:
  - Candlestick charts with volume
  - Moving averages (5-day, 10-day)
  - Daily returns distribution
  - Price vs Volume scatter plots

### 🤖 Prediction Models

#### LSTM (Long Short-Term Memory)
- **Deep Learning**: Neural network for time series prediction
- **Feature**: Automatic sequence generation
- **Metrics**: MSE, MAE for model evaluation
- **Future Predictions**: Next 5 days forecast

#### ARIMA (AutoRegressive Integrated Moving Average)
- **Statistical Model**: Time series forecasting
- **Auto Parameter Selection**: Automatic (p,d,q) optimization
- **Stationarity Testing**: ADF test with differencing
- **Confidence Intervals**: Prediction uncertainty quantification

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
./setup.sh

# Activate environment and start app
source stock_analysis_env/bin/activate
streamlit run streamlit_app.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv stock_analysis_env
source stock_analysis_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## 📋 Requirements

- Python 3.8 or higher
- Internet connection (for Yahoo Finance data)
- Web browser (for Streamlit interface)

### Python Dependencies
```
streamlit==1.28.1
yfinance==0.2.25
pandas==2.1.4
numpy==1.24.3
plotly==5.17.0
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
tensorflow==2.13.0
statsmodels==0.14.0
```

## 🎯 How to Use

### 1. **Stock Selection**
   - Choose from NVIDIA, OpenAI (Microsoft), or X (Meta)
   - Set your desired date range using the sidebar

### 2. **Load Data**
   - Click "Load Data" to fetch stock information
   - View the raw dataset with basic statistics

### 3. **Data Preprocessing**
   - Click "Preprocess Data" to clean the dataset
   - Review missing value handling and data validation results

### 4. **Analysis Configuration**
   - Select analysis period type:
     - **Specific Day**: Analyze a single trading day
     - **Specific Week**: Analyze a 7-day period
     - **Custom Range**: Define your own date range

### 5. **Run Analysis**
   - Click "Analyze Data" to generate insights
   - View metrics: price changes, volatility, volume
   - Identify significant movements and spikes

### 6. **Visualizations**
   - Interactive candlestick charts
   - Volume analysis
   - Moving averages
   - Returns distribution

### 7. **Predictions**
   - Choose between LSTM, ARIMA, or both models
   - Click "Train & Predict" to generate forecasts
   - View model performance metrics
   - Get next 5-day predictions

## 📊 Understanding the Results

### Analysis Metrics
- **Price Change**: Absolute and percentage change over period
- **Volatility**: Standard deviation of closing prices
- **Average Volume**: Mean trading volume
- **Price Range**: High and low price range

### Prediction Metrics
- **MSE (Mean Squared Error)**: Lower values indicate better predictions
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **Confidence Intervals**: Range of likely values (ARIMA only)

## 🛠️ Technical Details

### Data Processing Pipeline
1. **Data Fetching**: Yahoo Finance API via yfinance
2. **Validation**: Price logic and negative value checks
3. **Cleaning**: Missing value imputation and duplicate removal
4. **Feature Engineering**: Returns calculation and spike detection

### LSTM Architecture
- **Input Layer**: 60-day lookback sequences
- **LSTM Layers**: 2 layers with 50 units each
- **Dropout**: 0.2 rate for regularization
- **Output**: Single value prediction

### ARIMA Configuration
- **Parameter Search**: Automated (p,d,q) optimization
- **Stationarity**: ADF test with differencing
- **Validation**: Train/test split for performance evaluation

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all packages are installed
   pip install -r requirements.txt
   ```

2. **Data Loading Fails**
   - Check internet connection
   - Verify stock symbol availability
   - Try different date ranges

3. **Model Training Errors**
   - Ensure sufficient data points (minimum 70 for LSTM)
   - Check for data quality issues

4. **Memory Issues**
   - Reduce date range for large datasets
   - Close other applications if needed

### Performance Tips
- Use smaller date ranges for faster processing
- LSTM requires more computational resources than ARIMA
- Cache data to avoid repeated API calls

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new prediction models

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data API
- **Streamlit**: For the amazing web app framework
- **TensorFlow & Statsmodels**: For machine learning capabilities

---

**Happy Stock Analysis! 📈📊🤖**
