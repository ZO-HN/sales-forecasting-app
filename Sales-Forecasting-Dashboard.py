import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Metric card container */
.stMetric {
    background-color: #ffffff !important;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Metric label */
[data-testid="stMetricLabel"] {
    color: #333333 !important;
    font-size: 0.9rem;
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: #111111 !important;
    font-size: 1.8rem;
    font-weight: 700;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #2E86AB !important;
}

/* Headers */
h1, h2, h3, h4, h5 {
    color: #222222 !important;
}
</style>
""", unsafe_allow_html=True)



# Title
st.markdown('<p class="main-header">📊 Sales Forecasting Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/statistics.png", width=80)
    st.title("Dashboard Controls")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    st.markdown("---")
    st.subheader("⚙️ Forecast Settings")
    forecast_months = st.slider("Forecast Horizon (months)", 3, 12, 6)
    test_size = st.slider("Test Set Size (months)", 3, 12, 6)
    
    st.markdown("---")
    st.subheader("📈 Model Selection")
    models_to_run = st.multiselect(
        "Select Models",
        ["Moving Average", "Exponential Smoothing", "ARIMA", "SARIMA"],
        default=["Moving Average", "ARIMA", "SARIMA"]
    )

# Load and Process Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="latin-1")
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year_month'] = df['order_date'].dt.to_period('M')
    
    monthly_data = df.groupby('year_month').agg({
        'sales': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'profit': 'sum',
        'discount': 'mean'
    }).reset_index()
    
    monthly_data.columns = ['year_month', 'total_sales', 'avg_sales', 'num_orders',
                            'total_quantity', 'total_profit', 'avg_discount']
    monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()
    monthly_data = monthly_data.sort_values('date')
    
    ts_data = monthly_data.set_index('date')['total_sales']
    full_date_range = pd.date_range(start=ts_data.index.min(),
                                   end=ts_data.index.max(),
                                   freq='MS')
    ts_data = ts_data.reindex(full_date_range).fillna(method='ffill')
    
    return df, monthly_data, ts_data

if uploaded_file is not None:
    df, monthly_data, ts_data = load_data(uploaded_file)
    
    # Overview Section
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['order_date'].min().strftime('%Y-%m')} to {df['order_date'].max().strftime('%Y-%m')}")
    with col3:
        st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
    with col4:
        st.metric("Total Profit", f"${df['profit'].sum():,.0f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Exploratory Analysis", "📈 Time Series Analysis", 
                                             "🤖 Model Training", "🔮 Forecasting", "📑 Data Table"])
    
    with tab1:
        st.subheader("Monthly Sales Trends")
        
        # Calculate growth metrics
        monthly_data['mom_growth'] = monthly_data['total_sales'].pct_change() * 100
        monthly_data['avg_order_value'] = monthly_data['total_sales'] / monthly_data['num_orders']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Monthly Sales", f"${monthly_data['total_sales'].mean():,.0f}",
                     f"{monthly_data['mom_growth'].mean():.1f}% MoM avg")
        with col2:
            st.metric("Median Monthly Sales", f"${monthly_data['total_sales'].median():,.0f}")
        with col3:
            st.metric("Avg Orders/Month", f"{monthly_data['num_orders'].mean():.0f}")
        with col4:
            st.metric("Avg Order Value", f"${monthly_data['avg_order_value'].mean():.2f}")
        
        # Monthly Sales Line Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['total_sales'],
            mode='lines+markers',
            name='Monthly Sales',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Monthly Total Sales Trend",
            xaxis_title="Month",
            yaxis_title="Sales ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_orders = go.Figure()
            fig_orders.add_trace(go.Bar(
                x=monthly_data['date'],
                y=monthly_data['num_orders'],
                name='Number of Orders',
                marker_color='#A23B72'
            ))
            fig_orders.update_layout(
                title="Monthly Number of Orders",
                xaxis_title="Month",
                yaxis_title="Orders",
                height=350
            )
            st.plotly_chart(fig_orders, use_container_width=True)
        
        with col2:
            fig_profit = go.Figure()
            fig_profit.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['total_profit'],
                mode='lines+markers',
                name='Profit',
                line=dict(color='#F18F01', width=2),
                marker=dict(size=6)
            ))
            fig_profit.update_layout(
                title="Monthly Total Profit",
                xaxis_title="Month",
                yaxis_title="Profit ($)",
                height=350
            )
            st.plotly_chart(fig_profit, use_container_width=True)
        
        # MoM Growth
        fig_growth = go.Figure()
        colors = ['green' if x > 0 else 'red' for x in monthly_data['mom_growth']]
        fig_growth.add_trace(go.Bar(
            x=monthly_data['date'],
            y=monthly_data['mom_growth'],
            marker_color=colors,
            name='MoM Growth %'
        ))
        fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
        fig_growth.update_layout(
            title="Month-over-Month Sales Growth (%)",
            xaxis_title="Month",
            yaxis_title="Growth (%)",
            height=350
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with tab2:
        st.subheader("Time Series Decomposition & Analysis")
        
        # Stationarity Test
        st.markdown("#### 📊 Stationarity Test (Augmented Dickey-Fuller)")
        result = adfuller(ts_data.dropna())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ADF Statistic", f"{result[0]:.4f}")
        with col2:
            st.metric("p-value", f"{result[1]:.6f}")
        with col3:
            is_stationary = result[1] <= 0.05
            st.metric("Status", "✅ Stationary" if is_stationary else "❌ Non-Stationary")
        
        # Seasonal Decomposition
        if len(ts_data) >= 24:
            st.markdown("#### 📈 Seasonal Decomposition")

            decomposition = seasonal_decompose(
                ts_data,
                model='additive',
                period=12,
                extrapolate_trend='freq'
            )

            # Clean components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            resid = decomposition.resid.dropna()
            observed = decomposition.observed

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                vertical_spacing=0.04
            )

            fig.add_trace(
                go.Scatter(x=observed.index, y=observed, name="Observed"),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=trend.index, y=trend, name="Trend"),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=seasonal.index, y=seasonal, name="Seasonal"),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(x=resid.index, y=resid, name="Residual"),
                row=4, col=1
            )

            fig.update_layout(
                height=850,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            
            seasonal_strength = 1 - (decomposition.resid.var() / 
                                    (decomposition.resid + decomposition.seasonal).var())
            st.info(f"🔍 Seasonal Strength: {seasonal_strength:.4f} - " + 
                   ("Strong seasonality detected" if seasonal_strength > 0.6 else "Moderate seasonality"))
    
    with tab3:
        st.subheader("Model Training & Comparison")
        
        if len(models_to_run) == 0:
            st.warning("Please select at least one model from the sidebar.")
        else:
            # Train-Test Split
            train = ts_data[:-test_size]
            test = ts_data[-test_size:]
            
            st.info(f"📊 Training on {len(train)} months | Testing on {len(test)} months")
            
            results = {}
            predictions = {}
            
            with st.spinner("Training models..."):
                # Moving Average
                if "Moving Average" in models_to_run:
                    sma_pred = []
                    for i in range(len(test)):
                        if i == 0:
                            window_data = train[-3:]
                        else:
                            window_data = pd.concat([train[-3+i:], test[:i]])[-3:]
                        sma_pred.append(window_data.mean())
                    predictions['Moving Average'] = np.array(sma_pred)
                    results['Moving Average'] = {
                        'MAE': mean_absolute_error(test, sma_pred),
                        'RMSE': np.sqrt(mean_squared_error(test, sma_pred)),
                        'MAPE': np.mean(np.abs((test - sma_pred) / test)) * 100
                    }
                
                # Exponential Smoothing
                if "Exponential Smoothing" in models_to_run:
                    try:
                        es_model = ExponentialSmoothing(train, trend='add', seasonal=None)
                        es_fit = es_model.fit()
                        es_pred = es_fit.forecast(steps=len(test))
                        predictions['Exponential Smoothing'] = es_pred
                        results['Exponential Smoothing'] = {
                            'MAE': mean_absolute_error(test, es_pred),
                            'RMSE': np.sqrt(mean_squared_error(test, es_pred)),
                            'MAPE': np.mean(np.abs((test - es_pred) / test)) * 100
                        }
                    except Exception as e:
                        st.warning(f"Exponential Smoothing failed: {e}")
                
                # ARIMA
                if "ARIMA" in models_to_run:
                    try:
                        best_aic = np.inf
                        best_order = None
                        for p in range(0, 3):
                            for d in range(0, 2):
                                for q in range(0, 3):
                                    try:
                                        model = ARIMA(train, order=(p, d, q))
                                        fitted = model.fit()
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                    except:
                                        continue
                        
                        if best_order:
                            arima_model = ARIMA(train, order=best_order)
                            arima_fit = arima_model.fit()
                            arima_pred = arima_fit.forecast(steps=len(test))
                            predictions['ARIMA'] = arima_pred
                            results['ARIMA'] = {
                                'MAE': mean_absolute_error(test, arima_pred),
                                'RMSE': np.sqrt(mean_squared_error(test, arima_pred)),
                                'MAPE': np.mean(np.abs((test - arima_pred) / test)) * 100
                            }
                            st.session_state['best_arima_order'] = best_order
                    except Exception as e:
                        st.warning(f"ARIMA failed: {e}")
                
                # SARIMA
                if "SARIMA" in models_to_run and len(train) >= 24:
                    try:
                        sarima_model = SARIMAX(train, order=(1, 1, 1),
                                              seasonal_order=(1, 1, 1, 12))
                        sarima_fit = sarima_model.fit(disp=False)
                        sarima_pred = sarima_fit.forecast(steps=len(test))
                        predictions['SARIMA'] = sarima_pred
                        results['SARIMA'] = {
                            'MAE': mean_absolute_error(test, sarima_pred),
                            'RMSE': np.sqrt(mean_squared_error(test, sarima_pred)),
                            'MAPE': np.mean(np.abs((test - sarima_pred) / test)) * 100
                        }
                        st.session_state['sarima_fit'] = sarima_fit
                    except Exception as e:
                        st.warning(f"SARIMA failed: {e}")
            
            # Display Results
            if results:
                comparison_df = pd.DataFrame(results).T
                comparison_df = comparison_df.sort_values('MAPE')
                
                st.markdown("#### 🏆 Model Performance Comparison")
                st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen'), 
                           use_container_width=True)
                
                best_model = comparison_df.index[0]
                st.success(f"✅ Best Model: **{best_model}** (Lowest MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%)")
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train, name='Training Data',
                                        line=dict(color='#2E86AB', width=2)))
                fig.add_trace(go.Scatter(x=test.index, y=test, name='Actual Test',
                                        line=dict(color='green', width=2), mode='lines+markers'))
                
                colors = ['#FF6B6B', '#F18F01', '#6A4C93', '#A23B72']
                for i, (model_name, pred) in enumerate(predictions.items()):
                    fig.add_trace(go.Scatter(x=test.index, y=pred, name=f'{model_name}',
                                           line=dict(color=colors[i % len(colors)], width=2, dash='dash')))
                
                fig.update_layout(title="Model Predictions vs Actual", xaxis_title="Month",
                                yaxis_title="Sales ($)", hovermode='x unified', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state['best_model'] = best_model
                st.session_state['results'] = results
    
    with tab4:
        st.subheader("🔮 Future Sales Forecast")
        
        if 'best_model' in st.session_state:
            best_model = st.session_state['best_model']
            
            try:
                if best_model == 'SARIMA':
                    model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    fit = model.fit(disp=False)
                    forecast = fit.forecast(steps=forecast_months)
                    forecast_std = fit.get_forecast(steps=forecast_months).summary_frame()
                    
                elif best_model == 'ARIMA' and 'best_arima_order' in st.session_state:
                    model = ARIMA(ts_data, order=st.session_state['best_arima_order'])
                    fit = model.fit()
                    forecast = fit.forecast(steps=forecast_months)
                    forecast_std = fit.get_forecast(steps=forecast_months).summary_frame()
                    
                elif best_model == 'Exponential Smoothing':
                    model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
                    fit = model.fit()
                    forecast = fit.forecast(steps=forecast_months)
                    std_error = ts_data.std()
                    forecast_std = pd.DataFrame({
                        'mean_ci_lower': forecast - 1.96 * std_error,
                        'mean_ci_upper': forecast + 1.96 * std_error
                    })
                else:
                    forecast = [ts_data[-3:].mean()] * forecast_months
                    std_error = ts_data.std()
                    forecast_std = pd.DataFrame({
                        'mean_ci_lower': forecast - 1.96 * std_error,
                        'mean_ci_upper': forecast + 1.96 * std_error
                    })
                
                future_dates = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1),
                                            periods=forecast_months, freq='MS')
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Forecasted Sales", f"${forecast.sum():,.0f}")
                with col2:
                    st.metric("Avg Monthly Forecast", f"${forecast.mean():,.0f}")
                with col3:
                    growth = ((forecast.mean() - ts_data.mean()) / ts_data.mean()) * 100
                    st.metric("Projected Growth", f"{growth:+.2f}%")
                
                # Forecast Table
                forecast_df = pd.DataFrame({
                    'Month': future_dates.strftime('%Y-%m'),
                    'Forecasted Sales': forecast.values,
                    'Lower Bound (95%)': forecast_std['mean_ci_lower'].values,
                    'Upper Bound (95%)': forecast_std['mean_ci_upper'].values
                })
                st.dataframe(forecast_df.style.format({
                    'Forecasted Sales': '${:,.2f}',
                    'Lower Bound (95%)': '${:,.2f}',
                    'Upper Bound (95%)': '${:,.2f}'
                }), use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, name='Historical',
                                        line=dict(color='#2E86AB', width=2), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Forecast',
                                        line=dict(color='#FF6B6B', width=3, dash='dash'),
                                        mode='lines+markers', marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast_std['mean_ci_upper'],
                                        fill=None, mode='lines', line_color='rgba(255,107,107,0)',
                                        showlegend=False))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast_std['mean_ci_lower'],
                                        fill='tonexty', mode='lines', line_color='rgba(255,107,107,0)',
                                        name='95% Confidence', fillcolor='rgba(255,107,107,0.2)'))
                
                fig.add_vline(x=ts_data.index[-1], line_dash="dash", line_color="black",
                             annotation_text="Forecast Start", annotation_position="top")
                
                fig.update_layout(title=f"Sales Forecast - Next {forecast_months} Months ({best_model})",
                                xaxis_title="Month", yaxis_title="Sales ($)",
                                hovermode='x unified', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Forecast error: {e}")
        else:
            st.info("Please train models in the 'Model Training' tab first.")
    
    with tab5:
        st.subheader("📑 Monthly Data Table")
        st.dataframe(monthly_data.style.format({
            'total_sales': '${:,.2f}',
            'avg_sales': '${:,.2f}',
            'total_profit': '${:,.2f}',
            'avg_discount': '{:.2%}',
            'avg_order_value': '${:,.2f}',
            'mom_growth': '{:+.2f}%'
        }), use_container_width=True)
        
        # Download button
        csv = monthly_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Monthly Data",
            data=csv,
            file_name="monthly_sales_data.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Please upload your dataset CSV file using the sidebar to begin analysis.")
    st.markdown("""
    ### Expected Dataset Format:
    The CSV should contain columns including:
    - `order_date`: Date of order
    - `sales`: Sales amount
    - `quantity`: Quantity sold
    - `profit`: Profit amount
    - `discount`: Discount applied
    - Other relevant columns (region, category, etc.)
    """)
    
    # Show sample visualizations
    st.markdown("---")
    st.subheader("📊 Dashboard Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Exploratory Analysis**\n\nView trends, growth rates, and key performance metrics across time periods.")
    with col2:
        st.info("**Time Series Analysis**\n\nAnalyze seasonality, trends, and stationarity of your sales data.")
    with col3:
        st.info("**Forecasting Models**\n\nCompare multiple models and generate accurate sales predictions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sales Forecasting Dashboard</strong> | Built with Streamlit & Plotly</p>
    <p>Supports: Moving Average, Exponential Smoothing, ARIMA, SARIMA</p>
</div>
""", unsafe_allow_html=True)