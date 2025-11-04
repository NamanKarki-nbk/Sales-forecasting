import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SalesForecastingSystem:
    """
    Sales Forecasting System using Prophet for Store × Department combinations
    """
    
    def __init__(self, data_path):
        """
        Initialize the forecasting system
        
        Args:
            data_path: Path to the CSV file containing sales data
        """
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.models = {}
        self.forecasts = {}
        self.model_dir = 'models/forecast_saved_models'
        
        # Create directory for saved models
        os.makedirs(self.model_dir, exist_ok=True)
        
    def prepare_data_for_prophet(self, store_name, dept_name):
        """
        Prepare data for a specific store-department combination
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
            
        Returns:
            DataFrame formatted for Prophet (ds, y columns)
        """
        # Filter data for specific store and department
        filtered_df = self.df[
            (self.df['Store'] == store_name) & 
            (self.df['Dept'] == dept_name)
        ].copy()
        
        # Aggregate weekly sales by date (in case of duplicates)
        prophet_df = filtered_df.groupby('Date').agg({
            'Weekly_Sales': 'sum'
        }).reset_index()
        
        # Rename columns for Prophet
        prophet_df.columns = ['ds', 'y']
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        return prophet_df, filtered_df
    
    def create_holiday_dataframe(self, filtered_df):
        """
        Create holiday dataframe from the dataset
        
        Args:
            filtered_df: Filtered dataframe for store-dept combination
            
        Returns:
            Holiday dataframe for Prophet
        """
        holiday_df = filtered_df[filtered_df['IsHoliday'] == True][['Date', 'Holiday_Name']].copy()
        holiday_df = holiday_df.dropna(subset=['Holiday_Name'])
        
        if len(holiday_df) > 0:
            holiday_df.columns = ['ds', 'holiday']
            holiday_df = holiday_df.drop_duplicates()
            
            # Add lower and upper windows for holidays (effect before and after)
            holiday_df['lower_window'] = -2
            holiday_df['upper_window'] = 2
            
            return holiday_df
        
        return None
    
    def train_model(self, store_name, dept_name, forecast_periods=104):
        """
        Train Prophet model for a specific store-department combination
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
            forecast_periods: Number of periods to forecast (default: 52 weeks)
            
        Returns:
            model: Trained Prophet model
            forecast: Forecast dataframe
        """
        print(f"\nTraining model for: {store_name} - {dept_name}")
        
        # Prepare data
        prophet_df, filtered_df = self.prepare_data_for_prophet(store_name, dept_name)
        
        if len(prophet_df) < 2:
            print(f"Insufficient data for {store_name} - {dept_name}")
            return None, None
        
        # Create holiday dataframe
        holidays = self.create_holiday_dataframe(filtered_df)
        
        # Initialize Prophet model with custom parameters
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
            seasonality_mode='multiplicative',  # Better for sales data with varying amplitude
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,  # Strength of seasonality
            holidays_prior_scale=10.0,  # Strength of holiday effects
            interval_width=0.95  # Confidence interval
        )
        
        # Add custom seasonality for monthly patterns
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Add regressors (external factors)
        if 'Temperature' in filtered_df.columns:
            prophet_df = prophet_df.merge(
                filtered_df[['Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].drop_duplicates(),
                left_on='ds',
                right_on='Date',
                how='left'
            )
            prophet_df = prophet_df.drop('Date', axis=1)
            
            # Forward fill any missing values
            prophet_df = prophet_df.fillna(method='ffill').fillna(method='bfill')
            
            model.add_regressor('Temperature')
            model.add_regressor('Fuel_Price')
            model.add_regressor('CPI')
            model.add_regressor('Unemployment')
        
        # Fit the model
        model.fit(prophet_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_periods, freq='W')
        
        # Add regressors to future dataframe (use last known values)
        if 'Temperature' in prophet_df.columns:
            for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                # Use rolling average of last 12 weeks for future predictions
                last_values = prophet_df[col].tail(12).mean()
                future[col] = last_values
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Store model and forecast
        model_key = f"{store_name}_{dept_name}"
        self.models[model_key] = model
        self.forecasts[model_key] = forecast
        
        print(f"✓ Model trained successfully")
        print(f"  Historical data points: {len(prophet_df)}")
        print(f"  Forecast periods: {forecast_periods}")
        
        return model, forecast
    
    def save_model(self, store_name, dept_name):
        """
        Save trained model to disk
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
        """
        model_dir = 'models/forecast_saved_models'
        model_key = f"{store_name}_{dept_name}"
        
        if model_key not in self.models:
            print(f"No model found for {store_name} - {dept_name}")
            return
        
        # Create safe filename
        safe_filename = f"{store_name.replace(' ', '_')}_{dept_name.replace(' ', '_')}.pkl"
        filepath = os.path.join(model_dir, safe_filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_key], f)
        
        print(f"Model saved: {filepath}")
    
    def load_model(self, store_name, dept_name):
        """
        Load trained model from disk
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
            
        Returns:
            Loaded Prophet model
        """
        model_dir = 'models/forecast_saved_models'
        safe_filename = f"{store_name.replace(' ', '_')}_{dept_name.replace(' ', '_')}.pkl"
        filepath = os.path.join(model_dir, safe_filename)
        
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        model_key = f"{store_name}_{dept_name}"
        self.models[model_key] = model
        
        print(f"Model loaded: {filepath}")
        return model
    
    def train_all_combinations(self, stores=None, departments=None, forecast_periods=104):
        """
        Train models for all or selected store-department combinations
        
        Args:
            stores: List of store names (None = all stores)
            departments: List of department names (None = all departments)
            forecast_periods: Number of periods to forecast
        """
        if stores is None:
            stores = self.df['Store'].unique()
        
        if departments is None:
            departments = self.df['Dept'].unique()
        
        total = len(stores) * len(departments)
        current = 0
        
        print(f"Training {total} models...")
        print("=" * 60)
        
        results = []
        
        for store in stores:
            for dept in departments:
                current += 1
                print(f"\nProgress: {current}/{total}")
                
                try:
                    model, forecast = self.train_model(store, dept, forecast_periods)
                    
                    if model is not None:
                        self.save_model(store, dept)
                        results.append({
                            'store': store,
                            'department': dept,
                            'status': 'success'
                        })
                    else:
                        results.append({
                            'store': store,
                            'department': dept,
                            'status': 'insufficient_data'
                        })
                        
                except Exception as e:
                    print(f"✗ Error training model: {str(e)}")
                    results.append({
                        'store': store,
                        'department': dept,
                        'status': f'error: {str(e)}'
                    })
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        
        # Summary
        results_df = pd.DataFrame(results)
        print(f"\nSuccessful models: {len(results_df[results_df['status'] == 'success'])}")
        print(f"Failed models: {len(results_df[results_df['status'] != 'success'])}")
        
        return results_df
    
    # def get_forecast(self, store_name, dept_name, periods=104):
    #     """
    #     Get forecast for a specific store-department combination
        
    #     Args:
    #         store_name: Name of the store
    #         dept_name: Name of the department
    #         periods: Number of future periods to return
            
    #     Returns:
    #         DataFrame with forecast
    #     """
    #     model_key = f"{store_name}_{dept_name}"
        
    #     if model_key not in self.forecasts:
    #         print(f"No forecast found. Loading model and generating forecast...")
    #         model = self.load_model(store_name, dept_name)
    #         if model is None:
    #             return None
            
    #         _, forecast = self.train_model(store_name, dept_name)
        
    #     forecast_df = self.forecasts[model_key]
        
    #     # Get future predictions only
    #     future_forecast = forecast_df.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    #     future_forecast.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']
        
    #     return future_forecast
    def get_forecast(self, store_name, dept_name, periods=104):
        """
        Get forecast for a specific store-department combination
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
            periods: Number of future periods to return
            
        Returns:
            DataFrame with forecast
        """
        model_key = f"{store_name}_{dept_name}"

        # If forecast already exists in memory, return it
        if model_key in self.forecasts:
            forecast_df = self.forecasts[model_key]
        else:
            # Try loading model from disk
            model = self.load_model(store_name, dept_name)

            if model is None:
                # Model not found — train a new one
                print("Model not found. Training new model...")
                model, forecast_df = self.train_model(store_name, dept_name, forecast_periods=periods)
            else:
                # Model loaded — generate forecast without retraining
                prophet_df, _ = self.prepare_data_for_prophet(store_name, dept_name)
                future = model.make_future_dataframe(periods=periods, freq='W')

                # Ensure all regressors used during training exist
                regressors = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
                for col in regressors:
                    if col in prophet_df.columns:
                        # Use rolling average of last 12 weeks
                        future[col] = prophet_df[col].tail(12).mean()
                    else:
                        # If missing, fill with 0 (or some default)
                        future[col] = 0

                forecast_df = model.predict(future)
                self.forecasts[model_key] = forecast_df
                
        if 'Store' not in forecast_df.columns:
            forecast_df['Store'] = store_name
        if 'Dept' not in forecast_df.columns:
            forecast_df['Dept'] = dept_name   
        # Prepare output with proper column names
        
        future_forecast = forecast_df.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Store', 'Dept']]
        future_forecast.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound','Store', 'Dept']
        
        return future_forecast

    
    def get_model_performance(self, store_name, dept_name):
        """
        Calculate model performance metrics
        
        Args:
            store_name: Name of the store
            dept_name: Name of the department
            
        Returns:
            Dictionary with performance metrics
        """
        prophet_df, _ = self.prepare_data_for_prophet(store_name, dept_name)
        model_key = f"{store_name}_{dept_name}"
        
        if model_key not in self.forecasts:
            print("Model not found. Please train first.")
            return None
        
        forecast = self.forecasts[model_key]
        
        # Merge actual and predicted
        comparison = prophet_df.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
        
        # Calculate metrics
        mae = np.mean(np.abs(comparison['y'] - comparison['yhat']))
        mape = np.mean(np.abs((comparison['y'] - comparison['yhat']) / comparison['y'])) * 100
        rmse = np.sqrt(np.mean((comparison['y'] - comparison['yhat'])**2))
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Data_Points': len(comparison)
        }


# # Usage
# if __name__ == "__main__":
#     # Initialize the system
#     print("Sales Forecasting System with Prophet")
#     print("=" * 60)
    
#     # Load data
#     forecasting_system = SalesForecastingSystem('data/sales/synthetic_nepal_sales_with_more_holidays.csv')

#     #  Train ALL combinations 
#     print("\n\n### Example 3: Train All Models ###")
#     all_results = forecasting_system.train_all_combinations(forecast_periods=104)
#     all_results.to_csv('training_results.csv', index=False)