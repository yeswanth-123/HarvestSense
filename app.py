# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import joblib
# import datetime
# from datetime import timedelta
# import requests
# import os
# import openmeteo_requests
# import requests_cache
# from retry_requests import retry

# app = Flask(__name__)

# # --- Configuration ---
# DROPDOWN_DATA_CSV = 'AllCrops Weekly Features Without Dropping.csv'
# DF_FULL = None
# data_loaded = False

# # --- Discovered Column Lists ---
# COMMODITY_COLS = []
# VARIETY_COLS = []
# DISTRICT_COLS = []
# MARKET_COLS = []
# VALID_DISTRICTS_FROM_COORDS = set()
# COMMODITY_NAMES = []
# MODEL_CACHE = {}
# # ------------------------


# # --- District Coordinates Map ---
# # DISTRICT_COORDINATES = {
# # "Alipurduar": {"lat": 26.49, "lon": 89.57},
# # "Bankura": {"lat": 23.23, "lon": 87.07},
# # "Paschim Bardhaman": {"lat": 23.68, "lon": 87.01}, # Using Asansol
# # "Purba Bardhaman": {"lat": 23.27, "lon": 87.90}, # Using Barddhaman
# # "Birbhum": {"lat": 23.90, "lon": 87.53}, # Using Suri
# # "Cooch Behar": {"lat": 26.33, "lon": 89.48},
# # "Darjeeling": {"lat": 27.05, "lon": 88.27},
# # "Dakshin Dinajpur": {"lat": 25.23, "lon": 88.78}, # Using Balurghat
# # "Hooghly": {"lat": 22.88, "lon": 88.45}, # Using Chinsura
# # "Howrah": {"lat": 22.58, "lon": 88.38},
# # "Jalpaiguri": {"lat": 26.53, "lon": 88.77},
# # "Jhargram": {"lat": 22.45, "lon": 86.98},
# # "Kolkata": {"lat": 22.57, "lon": 88.36},
# # "Kalimpong": {"lat": 27.06, "lon": 88.47},
# # "Malda": {"lat": 25.00, "lon": 88.18}, # Using Ingraj Bazar
# # "Paschim Medinipur": {"lat": 22.42, "lon": 87.35}, # Using Medinipur
# # "Purba Medinipur": {"lat": 22.30, "lon": 87.97}, # Using Tamluk
# # "Murshidabad": {"lat": 24.10, "lon": 88.32}, # Using Baharampur
# # "Nadia": {"lat": 23.40, "lon": 88.55}, # Using Krishnanagar
# # "North 24 Parganas": {"lat": 22.72, "lon": 88.48}, # Using Barasat
# # "South 24 Parganas": {"lat": 22.53, "lon": 88.30}, # Using Alipore
# # "Purulia": {"lat": 23.33, "lon": 86.37},
# # "Uttar Dinajpur": {"lat": 25.62, "lon": 88.12} # Using Raiganj
# # }

# DISTRICT_COORDINATES = {
# "Alipurduar": {"lat": 26.49, "lon": 89.57},
# "Bankura": {"lat": 23.23, "lon": 87.07},
# "Paschim Bardhaman": {"lat": 23.68, "lon": 87.01}, # Using Asansol
# "Purba Bardhaman": {"lat": 23.27, "lon": 87.90}, # Using Barddhaman
# "Birbhum": {"lat": 23.90, "lon": 87.53}, # Using Suri
# "Coochbehar": {"lat": 26.33, "lon": 89.48},
# "Darjeeling": {"lat": 27.05, "lon": 88.27},
# "Dakshin_Dinajpur": {"lat": 25.23, "lon": 88.78}, # Using Balurghat
# "Hooghly": {"lat": 22.88, "lon": 88.45}, # Using Chinsura
# "Howrah": {"lat": 22.58, "lon": 88.38},
# "Jalpaiguri": {"lat": 26.53, "lon": 88.77},
# "Jhargram": {"lat": 22.45, "lon": 86.98},
# "Kolkata": {"lat": 22.57, "lon": 88.36},
# "Kalimpong": {"lat": 27.06, "lon": 88.47},
# "Malda": {"lat": 25.00, "lon": 88.18}, # Using Ingraj Bazar
# "Medinipur(W)": {"lat": 22.42, "lon": 87.35}, # Using Medinipur
# "Medinipur(E)": {"lat": 22.30, "lon": 87.97}, # Using Tamluk
# "Murshidabad": {"lat": 24.10, "lon": 88.32}, # Using Baharampur
# "Nadia": {"lat": 23.40, "lon": 88.55}, # Using Krishnanagar
# "North 24 Parganas": {"lat": 22.72, "lon": 88.48}, # Using Barasat
# "South 24 Parganas": {"lat": 22.53, "lon": 88.30}, # Using Alipore
# "Purulia": {"lat": 23.33, "lon": 86.37},
# "Uttar Dinajpur": {"lat": 25.62, "lon": 88.12} # Using Raiganj
# }
# # -------------------------------------

# def load_data(csv_path):
#     """Loads the full CSV and discovers dynamic columns for all dropdowns."""
#     global DF_FULL, data_loaded, COMMODITY_COLS, VARIETY_COLS, DISTRICT_COLS, \
#            MARKET_COLS, VALID_DISTRICTS_FROM_COORDS, COMMODITY_NAMES
    
#     if not os.path.exists(csv_path):
#         print(f"❌ FATAL ERROR: Dropdown CSV not found at '{csv_path}'. App cannot run.")
#         return False
    
#     try:
#         DF_FULL = pd.read_csv(csv_path)
        
#         COMMODITY_COLS = [col for col in DF_FULL.columns if col.startswith('Commodity_')]
#         VARIETY_COLS = [col for col in DF_FULL.columns if col.startswith('Variety_')]
#         DISTRICT_COLS = [col for col in DF_FULL.columns if col.startswith('District_')]
#         MARKET_COLS = [col for col in DF_FULL.columns if col.startswith('Market_')]
#         VALID_DISTRICTS_FROM_COORDS = set(DISTRICT_COORDINATES.keys())
        
#         # COMMODITY_NAMES = ["Green Chilli"] # This line is removed
#         COMMODITY_NAMES = sorted(list(set(col.split('_', 1)[1] for col in COMMODITY_COLS)))
        
#         print(f"✅ Dropdown data loaded. Found {len(COMMODITY_COLS)} Commodity, {len(VARIETY_COLS)} Variety, {len(DISTRICT_COLS)} District, {len(MARKET_COLS)} Market columns.")
#         print(f"✅ Serving {len(COMMODITY_NAMES)} total commodities: {COMMODITY_NAMES}")
#         data_loaded = True
#         return True
    
#     except Exception as e:
#         print(f"❌ Error loading dropdown data: {e}")
#         return False

# # Load data at startup
# data_loaded = load_data(DROPDOWN_DATA_CSV)

# # --- Open-Meteo API Client ---
# cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
# retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
# openmeteo = openmeteo_requests.Client(session = retry_session)
# # -------------------------------


# def get_most_recent_price(district, variety, commodity, market):
#     """Finds the most recent 'Current_Week_Mean_Price' row from the CSV."""
#     if DF_FULL is None:
#         return None, (None, "CSV data not loaded.")
    
#     district_col = f"District_{district}"
#     variety_col = f"Variety_{variety}"
#     market_col = f"Market_{market}"
#     filters = []
    
#     # Commodity Filter
#     if f"Commodity_{commodity}" in DF_FULL.columns:
#         filters.append((DF_FULL[f"Commodity_{commodity}"] == 1))
#     else:
#         return None, (None, f"Commodity column for '{commodity}' not found.")
    
#     # Variety Filter
#     if variety_col in DF_FULL.columns:
#         filters.append((DF_FULL[variety_col] == 1))
#     else:
#         return None, (None, f"Variety column '{variety_col}' not found.")
    
#     # District Filter
#     if district_col in DF_FULL.columns:
#         filters.append((DF_FULL[district_col] == 1))
#     else:
#         return None, (None, f"District column '{district_col}' not found.")

#     # Market Filter
#     if market_col in DF_FULL.columns:
#         filters.append((DF_FULL[market_col] == 1))
#     else:
#         return None, (None, f"Market column '{market_col}' not found.")
    
#     try:
#         combined_filter = pd.Series(True, index=DF_FULL.index)
#         for f in filters:
#             combined_filter = combined_filter & f
        
#         filtered_df = DF_FULL[combined_filter]
        
#         if filtered_df.empty:
#             return None, (None, f"No data found for combination: {commodity}, {district}, {variety}, {market}.")
        
#         last_row_series = filtered_df.iloc[-1]
#         last_row_dict = last_row_series.to_dict()
#         price = last_row_dict['Current_Week_Mean_Price']
        
#         return last_row_dict, (float(price), None) # Return the full row AND the price
#     except Exception as e:
#         return None, (None, f"Error during price lookup: {str(e)}")

# # --- Predictor Class ---
# class CommodityPricePredictor:
#     def __init__(self):
#         self.model_data = None
#         self.model = None
#         self.scaler = None
#         self.imputer = None
#         self.feature_names = None
    
#     def load_model(self, model_path):
#         try:
#             self.model_data = joblib.load(model_path)
#             self.model = self.model_data['model']
#             self.scaler = self.model_data['scaler']
#             self.imputer = self.model_data['imputer']
#             self.feature_names = self.model_data['feature_names']
#             print(f"✅ Model '{model_path}' loaded successfully")
#             print(f"✅ Model trained on {len(self.feature_names)} features.")
#             return True
#         except Exception as e:
#             print(f"❌ Error loading model '{model_path}': {e}")
#             return False
    
#     def get_weather_forecast(self, district="Kolkata"):
#         coords = DISTRICT_COORDINATES.get(district)
#         if not coords:
#             print(f"⚠️ Warning: District '{district}' not found. Defaulting to Kolkata.")
#             coords = DISTRICT_COORDINATES["Kolkata"]
#         url = "https://api.open-meteo.com/v1/forecast"
#         params = {"latitude": coords["lat"], "longitude": coords["lon"],
#                   "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
#                             "rain_sum", "relative_humidity_2m_mean", "sunshine_duration", "wind_speed_10m_mean"],
#                   "timezone": "auto", "forecast_days": 14}
#         try:
#             responses = openmeteo.weather_api(url, params=params)
#             response = responses[0]
#             daily = response.Daily()
#             daily_data = {"date": pd.date_range(
#                 start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
#                 end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
#                 freq = pd.Timedelta(seconds = daily.Interval()),
#                 inclusive = "left"
#             )}
#             daily_data["temperature_2m_mean"] = daily.Variables(0).ValuesAsNumpy()
#             daily_data["temperature_2m_max"] = daily.Variables(1).ValuesAsNumpy()
#             daily_data["temperature_2m_min"] = daily.Variables(2).ValuesAsNumpy()
#             daily_data["rain_sum"] = daily.Variables(3).ValuesAsNumpy()
#             daily_data["relative_humidity_2m_mean"] = daily.Variables(4).ValuesAsNumpy()
#             daily_data["sunshine_duration"] = daily.Variables(5).ValuesAsNumpy()
#             daily_data["wind_speed_10m_mean"] = daily.Variables(6).ValuesAsNumpy()
            
#             daily_dataframe = pd.DataFrame(data = daily_data)
#             forecasts = []
#             for _, row in daily_dataframe.iterrows():
#                 forecasts.append({'date': row['date'].strftime('%Y-%m-%d'),
#                                   'temp_mean': float(round(row['temperature_2m_mean'], 1)),
#                                   'temp_min': float(round(row['temperature_2m_min'], 1)),
#                                   'temp_max': float(round(row['temperature_2m_max'], 1)),
#                                   'rain': float(round(row['rain_sum'], 1)),
#                                   'humidity': int(round(row['relative_humidity_2m_mean'])),
#                                   'wind_speed': float(round(row['wind_speed_10m_mean'], 1)),
#                                   'sunshine': int(row['sunshine_duration']),
#                                   'description': self.get_weather_description(row['rain_sum'])})
#             return forecasts[:14]
#         except Exception as e:
#             print(f"❌ Weather API error: {e}")
#             return self.create_fallback_weather()
    
#     def get_historical_weather(self, district="Kolkata"):
#         coords = DISTRICT_COORDINATES.get(district)
#         if not coords:
#             print(f"⚠️ Warning: District '{district}' not found. Defaulting to Kolkata for historical data.")
#             coords = DISTRICT_COORDINATES["Kolkata"]
        
#         end_date = datetime.date.today() - timedelta(days=1)
#         start_date = end_date - timedelta(days=20)
        
#         url = "https://archive-api.open-meteo.com/v1/archive"
#         params = {
#             "latitude": coords["lat"],
#             "longitude": coords["lon"],
#             "start_date": start_date.strftime("%Y-%m-%d"),
#             "end_date": end_date.strftime("%Y-%m-%d"),
#             "daily": ["temperature_2m_mean", "rain_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean", "sunshine_duration"],
#             "timezone": "auto"
#         }
        
#         try:
#             print(f"Fetching live historical data for {district} from {start_date} to {end_date}...")
#             responses = openmeteo.weather_api(url, params=params)
#             response = responses[0]
            
#             daily = response.Daily()
#             daily_data = {
#                 "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
#                 "rain_sum": daily.Variables(1).ValuesAsNumpy(),
#                 "relative_humidity_2m_mean": daily.Variables(2).ValuesAsNumpy(),
#                 "wind_speed_10m_mean": daily.Variables(3).ValuesAsNumpy(),
#                 "sunshine_duration": daily.Variables(4).ValuesAsNumpy()
#             }
            
#             df = pd.DataFrame(data=daily_data)
            
#             aggregated_data = {
#                 'Past_3W_Temp_Mean': np.nanmean(df['temperature_2m_mean']),
#                 'Past_3W_Rain_Sum': np.nansum(df['rain_sum']),
#                 'Past_3W_Humidity_Mean': np.nanmean(df['relative_humidity_2m_mean']),
#                 'Past_3W_Wind_Mean': np.nanmean(df['wind_speed_10m_mean']),
#                 'Past_3W_Sunshine_Sum': np.nansum(df['sunshine_duration']),
#                 'Past_3W_Temp_Std': np.nanstd(df['temperature_2m_mean'])
#             }
#             print(f"✅ Live historical data aggregated: TempMean {aggregated_data['Past_3W_Temp_Mean']:.1f}C, RainSum {aggregated_data['Past_3W_Rain_Sum']:.1f}mm")
#             return aggregated_data
        
#         except Exception as e:
#             print(f"❌ Historical Weather API error: {e}. Will fall back to CSV data.")
#             return None
    
#     def get_weather_description(self, rain):
#         if rain == 0: return "Clear sky"
#         elif rain < 5: return "Light rain"
#         elif rain < 15: return "Moderate rain"
#         else: return "Heavy rain"
    
#     def create_fallback_weather(self):
#         print("❌ API FAILED. Using dummy fallback weather.")
#         forecasts = []
#         base_date = datetime.datetime.now()
#         for i in range(14):
#             date = base_date + timedelta(days=i + 1)
#             temp_mean = 26 + np.random.uniform(-2, 2)
#             rain = max(0, np.random.normal(1, 2))
#             forecasts.append({'date': date.strftime('%Y-%m-%d'),
#                               'temp_mean': float(round(temp_mean, 1)),
#                               'temp_min': float(round(temp_mean - 3, 1)),
#                               'temp_max': float(round(temp_mean + 4, 1)),
#                               'rain': float(round(rain, 1)),
#                               'humidity': int(75 + np.random.randint(-10, 10)),
#                               'wind_speed': float(round(3 + np.random.uniform(0, 2), 1)),
#                               'sunshine': int(60000 + np.random.randint(0, 20000)),
#                               'description': self.get_weather_description(rain)})
#         return forecasts
    
#     def create_prediction_features(self, current_price, weather_forecast, day_index, start_price_row, live_historical_data_override):
#         weather = weather_forecast[day_index]
#         date_obj = datetime.datetime.strptime(weather['date'], '%Y-%m-%d')
#         features = {}
        
#         hist_data_source = live_historical_data_override if live_historical_data_override is not None else start_price_row
        
#         features['Current_Week_Min_Price'] = current_price * 0.95
#         features['Current_Week_Max_Price'] = current_price * 1.05
#         features['Current_Week_Mean_Price'] = current_price
#         features['Current_Week_Std_Price'] = current_price * 0.03
#         features['Price_Range_Week'] = current_price * 0.1
        
#         features['Forecast_Temp_Mean_Next_Week'] = weather['temp_mean']
#         features['Forecast_Temp_Min_Next_Week'] = weather['temp_min']
#         features['Forecast_Temp_Max_Next_Week'] = weather['temp_max']
#         features['Forecast_Rain_Sum_Next_Week'] = weather['rain']
#         features['Forecast_Humidity_Mean_Next_Week'] = weather['humidity']
#         features['Forecast_Wind_Mean_Next_Week'] = weather['wind_speed']
#         features['Forecast_Sunshine_Sum_Next_Week'] = weather['sunshine']
        
#         features['Rolling_3W_Mean'] = start_price_row.get('Rolling_3W_Mean', current_price)
#         features['Rolling_3W_Std'] = start_price_row.get('Rolling_3W_Std', current_price * 0.04)
#         features['Pct_Change_Last_Week'] = start_price_row.get('Pct_Change_Last_Week', 0)
#         features['Price_Trend_Slope_3W'] = start_price_row.get('Price_Trend_Slope_3W', 0)
#         features['Price_Velocity'] = start_price_row.get('Price_Velocity', 0)
#         features['Price_Acceleration'] = start_price_row.get('Price_Acceleration', 0)
#         features['Momentum_Strength'] = start_price_row.get('Momentum_Strength', 0.5)
#         features['Price_Position_Ratio'] = start_price_row.get('Price_Position_Ratio', 0.5)
        
#         features['Next_Week_Month'] = date_obj.month
#         features['Is_Monsoon_Next_Week'] = 1 if 6 <= date_obj.month <= 9 else 0
#         features['Is_Harvest_Season_Next_Week'] = 1 if date_obj.month in [10, 11] else 0
#         features['Is_Festive_Week_Next_Week'] = 0
        
#         features['Heat_Stress_Days_Next_Week'] = 1 if weather['temp_max'] > 35 else 0
#         features['Cold_Stress_Days_Next_Week'] = 1 if weather['temp_min'] < 15 else 0
#         features['Heavy_Rain_Indicator_Next_Week'] = 1 if weather['rain'] > 20 else 0
#         features['Drought_Indicator_Next_Week'] = 1 if weather['rain'] < 1 and weather['temp_max'] > 32 else 0
        
#         features['Past_3W_Temp_Mean'] = hist_data_source.get('Past_3W_Temp_Mean', 0)
#         features['Past_3W_Rain_Sum'] = hist_data_source.get('Past_3W_Rain_Sum', 0)
#         features['Past_3W_Humidity_Mean'] = hist_data_source.get('Past_3W_Humidity_Mean', 0)
#         features['Past_3W_Wind_Mean'] = hist_data_source.get('Past_3W_Wind_Mean', 0)
#         features['Past_3W_Sunshine_Sum'] = hist_data_source.get('Past_3W_Sunshine_Sum', 0)
#         features['Past_3W_Temp_Std'] = hist_data_source.get('Past_3W_Temp_Std', 0)
        
#         features['Price_Anomaly'] = current_price - features['Rolling_3W_Mean']
#         features['Heat_x_Drought'] = features['Forecast_Temp_Max_Next_Week'] * features['Drought_Indicator_Next_Week']
#         features['Rain_x_Monsoon'] = features['Forecast_Rain_Sum_Next_Week'] * features['Is_Monsoon_Next_Week']
        
#         features['Temp_Anomaly'] = hist_data_source.get('Temp_Anomaly', 0)
#         features['Rain_Anomaly'] = hist_data_source.get('Rain_Anomaly', 0)
#         features['Volatility_Rain_Interaction'] = hist_data_source.get('Volatility_Rain_Interaction', 0)
#         features['Temp_Volatility_Next_Week'] = 0
#         features['Humidity_Volatility_Next_Week'] = 0
        
#         for feature in self.feature_names:
#             if feature not in features:
#                 features[feature] = 0
        
#         return features
    
#     def predict_price(self, features):
#         try:
#             # --- START: FIX FOR UserWarning ---
#             input_df = pd.DataFrame([features])
#             input_df_ordered = input_df.reindex(columns=self.feature_names)
#             input_imputed = self.imputer.transform(input_df_ordered)
#             input_scaled = self.scaler.transform(input_imputed)
#             # --- END OF FIX ---

#             prediction = self.model.predict(input_scaled)[0]
#             prediction_float = float(prediction)
            
#             return max(100, min(20000, prediction_float))
        
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             missing_keys = [f for f in self.feature_names if f not in features]
#             if missing_keys:
#                 print(f"Missing keys in features dict: {missing_keys[:5]}")
#             return features['Current_Week_Mean_Price'] * (1 + np.random.uniform(-0.05, 0.05))


# # --- Specialist Model Loader ---
# def get_predictor(commodity):
#     if commodity in MODEL_CACHE:
#         print(f"Returning cached model for {commodity}.")
#         return MODEL_CACHE[commodity]
    
#     model_filename = f"trained_{commodity.lower().replace(' ', '_')}_model.pkl"
#     if not os.path.exists(model_filename):
#         print(f"❌ ERROR: Model file not found: {model_filename}")
#         return None
    
#     try:
#         print(f"Loading specialist model from {model_filename}...")
#         predictor = CommodityPricePredictor()
#         if predictor.load_model(model_filename):
#             MODEL_CACHE[commodity] = predictor
#             return predictor
#         else:
#             return None
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load model {model_filename}: {e}")
#         return None

# # --- Flask Routes ---

# @app.route('/')
# def home():
#     if not data_loaded:
#         return "Error: Could not load dropdown data CSV file. Please check server logs.", 500
#     return render_template('index.html', commodities=COMMODITY_NAMES)

# # --- API: Get Varieties ---
# @app.route('/api/get-varieties', methods=['POST'])
# def get_varieties():
#     if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
#     data = request.json
#     commodity = data.get('commodity')
    
#     filtered_df = DF_FULL
    
#     if f"Commodity_{commodity}" in DF_FULL.columns:
#         filtered_df = DF_FULL[DF_FULL[f"Commodity_{commodity}"] == 1]
#     else:
#         return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
#     varieties = get_options_from_df(filtered_df, 'Variety_')
#     return jsonify({'success': True, 'varieties': varieties})

# # --- API: Get Districts ---
# @app.route('/api/get-districts', methods=['POST'])
# def get_districts():
#     if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
#     data = request.json
#     commodity = data.get('commodity')
#     variety = data.get('variety')
    
#     filtered_df = DF_FULL
    
#     if f"Commodity_{commodity}" in DF_FULL.columns:
#         filtered_df = filtered_df[filtered_df[f"Commodity_{commodity}"] == 1]
#     else:
#         return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
#     variety_col = f"Variety_{variety}"
#     if variety_col not in DF_FULL.columns:
#         return jsonify({'success': False, 'error': f"Invalid variety: {variety}"})
    
#     filtered_df = filtered_df[filtered_df[variety_col] == 1]
    
#     districts_from_data = get_options_from_df(filtered_df, 'District_')
#     valid_districts = [d for d in districts_from_data if d in VALID_DISTRICTS_FROM_COORDS]
    
#     return jsonify({'success': True, 'districts': valid_districts})

# # --- API: Get Markets ---
# @app.route('/api/get-markets', methods=['POST'])
# def get_markets():
#     if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
#     data = request.json
#     commodity = data.get('commodity')
#     variety = data.get('variety')
#     district = data.get('district')
    
#     filtered_df = DF_FULL
    
#     # Filter by Commodity
#     if f"Commodity_{commodity}" in DF_FULL.columns:
#         filtered_df = filtered_df[filtered_df[f"Commodity_{commodity}"] == 1]
#     else:
#         return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
#     # Filter by Variety
#     variety_col = f"Variety_{variety}"
#     if variety_col not in DF_FULL.columns:
#         return jsonify({'success': False, 'error': f"Invalid variety: {variety}"})
#     filtered_df = filtered_df[filtered_df[variety_col] == 1]

#     # Filter by District
#     district_col = f"District_{district}"
#     if district_col not in DF_FULL.columns:
#         return jsonify({'success': False, 'error': f"Invalid district: {district}"})
#     filtered_df = filtered_df[filtered_df[district_col] == 1]
    
#     # --- START OF FIX ---
#     # Get available markets, but ignore "Activity" or other non-market columns
#     ignore_list = ['Activity'] 
#     markets = get_options_from_df(filtered_df, 'Market_', ignore_list=ignore_list)
#     # --- END OF FIX ---
    
#     return jsonify({'success': True, 'markets': markets})

# # --- Helper function for dropdown APIs ---
# # --- START OF FIX ---
# # Updated function to accept an ignore_list
# def get_options_from_df(df, prefix, ignore_list=None):
#     if ignore_list is None:
#         ignore_list = []
        
#     cols = [col for col in df.columns if col.startswith(prefix)]
#     options = []
#     for col in cols:
#         if df[col].any():
#             option_name = col.split('_', 1)[1]
#             if option_name not in ignore_list: # Check against the ignore list
#                 options.append(option_name)
#     return sorted(list(set(options)))
# # --- END OF FIX ---

# # --- /api/predict Route ---
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         district = data.get('district')
#         commodity = data.get('commodity')
#         variety = data.get('variety')
#         market = data.get('market')
        
#         if not district or not variety or not commodity or not market:
#             return jsonify({'success': False, 'error': 'Missing input: Commodity, Variety, District, and Market are required.'})
        
#         predictor = get_predictor(commodity)
#         if predictor is None:
#             return jsonify({'success': False, 'error': f"No model available for '{commodity}'. Please train it first."})
        
#         live_historical_data = predictor.get_historical_weather(district)
        
#         historical_row, (start_price, error) = get_most_recent_price( district, variety, commodity, market)
#         if error:
#             return jsonify({'success': False, 'error': error})
        
#         weather_forecasts = predictor.get_weather_forecast(district)
#         if not weather_forecasts:
#             return jsonify({'success': False, 'error': 'Could not retrieve weather data.'})
        
#         predictions = []
#         last_price = start_price
#         current_historical_row = historical_row.copy()
        
#         for i in range(min(14, len(weather_forecasts))):
#             hist_data_for_loop = live_historical_data if i == 0 else None
            
#             features = predictor.create_prediction_features(
#                 last_price,
#                 weather_forecasts,
#                 i,
#                 current_historical_row,
#                 hist_data_for_loop
#             )
            
#             predicted_price = predictor.predict_price(features)
            
#             predictions.append({
#                 'date': weather_forecasts[i]['date'],
#                 'predicted_price': round(predicted_price, 2),
#                 'weather': weather_forecasts[i],
#                 'change_percent': round(((predicted_price - start_price) / start_price) * 100, 2),
#                 'type': 'Predicted'
#             })
            
#             last_price = predicted_price
            
#             new_historical_row = current_historical_row.copy()
#             lag_1w = current_historical_row.get('Current_Week_Mean_Price', last_price)
#             lag_2w = current_historical_row.get('Lag_1W_Price', last_price)
            
#             new_historical_row['Rolling_3W_Mean'] = (lag_1w + lag_2w + last_price) / 3
#             new_historical_row['Pct_Change_Last_Week'] = (last_price - lag_1w) / (lag_1w + 1e-6)
#             new_historical_row['Current_Week_Mean_Price'] = last_price
#             new_historical_row['Lag_1W_Price'] = lag_1w
#             new_historical_row['Lag_2W_Price'] = lag_2w
            
#             current_historical_row = new_historical_row
        
#         return jsonify({
#             'success': True,
#             'start_price': start_price,
#             'district': district,
#             'commodity': commodity,
#             'variety': variety,
#             'market': market,
#             'predicted_data': predictions
#         })
    
#     except Exception as e:
#         print(f"Error in /api/predict: {e}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({'success': False, 'error': str(e)})

# if __name__ == '__main__':
#     if data_loaded:
#         print("🚀 Starting Specialist Commodity Price Predictor")
#         print("📈 Dropdown data loaded. Models will be loaded on-demand.")
#         print("🌦️ Using Open-Meteo for live weather data.")
#         print("🌐 Open: http://localhost:5000")
#         app.run(debug=True, host='0.0.0.0', port=5000)
#     else:
#         print("❌ Application failed to start. Please fix the CSV file path.")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
from datetime import timedelta
import requests
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry

app = Flask(__name__)

# --- Configuration ---
DROPDOWN_DATA_CSV = 'AllCrops Weekly Features Without Dropping.csv'
DF_FULL = None
data_loaded = False

# --- Discovered Column Lists ---
COMMODITY_COLS = []
VARIETY_COLS = []
DISTRICT_COLS = []
MARKET_COLS = []
VALID_DISTRICTS_FROM_COORDS = set()
COMMODITY_NAMES = []
MODEL_CACHE = {}
# ------------------------


# --- District Coordinates Map ---
DISTRICT_COORDINATES = {
"Alipurduar": {"lat": 26.49, "lon": 89.57},
"Bankura": {"lat": 23.23, "lon": 87.07},
"Paschim Bardhaman": {"lat": 23.68, "lon": 87.01}, # Using Asansol
"Purba Bardhaman": {"lat": 23.27, "lon": 87.90}, # Using Barddhaman
"Birbhum": {"lat": 23.90, "lon": 87.53}, # Using Suri
"Coochbehar": {"lat": 26.33, "lon": 89.48},
"Darjeeling": {"lat": 27.05, "lon": 88.27},
"Dakshin_Dinajpur": {"lat": 25.23, "lon": 88.78}, # Using Balurghat
"Hooghly": {"lat": 22.88, "lon": 88.45}, # Using Chinsura
"Howrah": {"lat": 22.58, "lon": 88.38},
"Jalpaiguri": {"lat": 26.53, "lon": 88.77},
"Jhargram": {"lat": 22.45, "lon": 86.98},
"Kolkata": {"lat": 22.57, "lon": 88.36},
"Kalimpong": {"lat": 27.06, "lon": 88.47},
"Malda": {"lat": 25.00, "lon": 88.18}, # Using Ingraj Bazar
"Medinipur(W)": {"lat": 22.42, "lon": 87.35}, # Using Medinipur
"Medinipur(E)": {"lat": 22.30, "lon": 87.97}, # Using Tamluk
"Murshidabad": {"lat": 24.10, "lon": 88.32}, # Using Baharampur
"Nadia": {"lat": 23.40, "lon": 88.55}, # Using Krishnanagar
"North 24 Parganas": {"lat": 22.72, "lon": 88.48}, # Using Barasat
"South 24 Parganas": {"lat": 22.53, "lon": 88.30}, # Using Alipore
"Purulia": {"lat": 23.33, "lon": 86.37},
"Uttar Dinajpur": {"lat": 25.62, "lon": 88.12} # Using Raiganj
}
# -------------------------------------

def load_data(csv_path):
    """Loads the full CSV and discovers dynamic columns for all dropdowns."""
    global DF_FULL, data_loaded, COMMODITY_COLS, VARIETY_COLS, DISTRICT_COLS, \
           MARKET_COLS, VALID_DISTRICTS_FROM_COORDS, COMMODITY_NAMES
    
    if not os.path.exists(csv_path):
        print(f"❌ FATAL ERROR: Dropdown CSV not found at '{csv_path}'. App cannot run.")
        return False
    
    try:
        DF_FULL = pd.read_csv(csv_path)
        
        COMMODITY_COLS = [col for col in DF_FULL.columns if col.startswith('Commodity_')]
        VARIETY_COLS = [col for col in DF_FULL.columns if col.startswith('Variety_')]
        DISTRICT_COLS = [col for col in DF_FULL.columns if col.startswith('District_')]
        MARKET_COLS = [col for col in DF_FULL.columns if col.startswith('Market_')]
        VALID_DISTRICTS_FROM_COORDS = set(DISTRICT_COORDINATES.keys())
        
        COMMODITY_NAMES = sorted(list(set(col.split('_', 1)[1] for col in COMMODITY_COLS)))
        
        print(f"✅ Dropdown data loaded. Found {len(COMMODITY_COLS)} Commodity, {len(VARIETY_COLS)} Variety, {len(DISTRICT_COLS)} District, {len(MARKET_COLS)} Market columns.")
        print(f"✅ Serving {len(COMMODITY_NAMES)} total commodities: {COMMODITY_NAMES}")
        data_loaded = True
        return True
    
    except Exception as e:
        print(f"❌ Error loading dropdown data: {e}")
        return False

# Load data at startup
data_loaded = load_data(DROPDOWN_DATA_CSV)

# --- Open-Meteo API Client ---
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
# -------------------------------


def get_most_recent_price(district, variety, commodity, market):
    """Finds the most recent 'Current_Week_Mean_Price' row from the CSV."""
    if DF_FULL is None:
        return None, (None, "CSV data not loaded.")
    
    district_col = f"District_{district}"
    variety_col = f"Variety_{variety}"
    market_col = f"Market_{market}"
    filters = []
    
    # Commodity Filter
    if f"Commodity_{commodity}" in DF_FULL.columns:
        filters.append((DF_FULL[f"Commodity_{commodity}"] == 1))
    else:
        return None, (None, f"Commodity column for '{commodity}' not found.")
    
    # Variety Filter
    if variety_col in DF_FULL.columns:
        filters.append((DF_FULL[variety_col] == 1))
    else:
        return None, (None, f"Variety column '{variety_col}' not found.")
    
    # District Filter
    if district_col in DF_FULL.columns:
        filters.append((DF_FULL[district_col] == 1))
    else:
        return None, (None, f"District column '{district_col}' not found.")

    # Market Filter
    if market_col in DF_FULL.columns:
        filters.append((DF_FULL[market_col] == 1))
    else:
        return None, (None, f"Market column '{market_col}' not found.")
    
    try:
        combined_filter = pd.Series(True, index=DF_FULL.index)
        for f in filters:
            combined_filter = combined_filter & f
        
        filtered_df = DF_FULL[combined_filter]
        
        if filtered_df.empty:
            return None, (None, f"No data found for combination: {commodity}, {district}, {variety}, {market}.")
        
        last_row_series = filtered_df.iloc[-1]
        last_row_dict = last_row_series.to_dict()
        price = last_row_dict['Current_Week_Mean_Price']
        
        return last_row_dict, (float(price), None) # Return the full row AND the price
    except Exception as e:
        return None, (None, f"Error during price lookup: {str(e)}")

# --- Predictor Class ---
class CommodityPricePredictor:
    def __init__(self):
        self.model_data = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
    
    def load_model(self, model_path):
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.imputer = self.model_data['imputer']
            self.feature_names = self.model_data['feature_names']
            print(f"✅ Model '{model_path}' loaded successfully")
            print(f"✅ Model trained on {len(self.feature_names)} features.")
            return True
        except Exception as e:
            print(f"❌ Error loading model '{model_path}': {e}")
            return False
    
    def get_weather_forecast(self, district="Kolkata"):
        coords = DISTRICT_COORDINATES.get(district)
        if not coords:
            print(f"⚠️ Warning: District '{district}' not found. Defaulting to Kolkata.")
            coords = DISTRICT_COORDINATES["Kolkata"]
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": coords["lat"], "longitude": coords["lon"],
                  "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                            "rain_sum", "relative_humidity_2m_mean", "sunshine_duration", "wind_speed_10m_mean"],
                  "timezone": "auto", "forecast_days": 14}
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}
            daily_data["temperature_2m_mean"] = daily.Variables(0).ValuesAsNumpy()
            daily_data["temperature_2m_max"] = daily.Variables(1).ValuesAsNumpy()
            daily_data["temperature_2m_min"] = daily.Variables(2).ValuesAsNumpy()
            daily_data["rain_sum"] = daily.Variables(3).ValuesAsNumpy()
            daily_data["relative_humidity_2m_mean"] = daily.Variables(4).ValuesAsNumpy()
            daily_data["sunshine_duration"] = daily.Variables(5).ValuesAsNumpy()
            daily_data["wind_speed_10m_mean"] = daily.Variables(6).ValuesAsNumpy()
            
            daily_dataframe = pd.DataFrame(data = daily_data)
            forecasts = []
            for _, row in daily_dataframe.iterrows():
                forecasts.append({'date': row['date'].strftime('%d-%m-%Y'), # <-- 1. DATE FORMAT FIX
                                  'temp_mean': float(round(row['temperature_2m_mean'], 1)),
                                  'temp_min': float(round(row['temperature_2m_min'], 1)),
                                  'temp_max': float(round(row['temperature_2m_max'], 1)),
                                  'rain': float(round(row['rain_sum'], 1)),
                                  'humidity': int(round(row['relative_humidity_2m_mean'])),
                                  'wind_speed': float(round(row['wind_speed_10m_mean'], 1)),
                                  'sunshine': int(row['sunshine_duration']),
                                  'description': self.get_weather_description(row['rain_sum'])})
            return forecasts[:14]
        except Exception as e:
            print(f"❌ Weather API error: {e}")
            return self.create_fallback_weather()
    
    def get_historical_weather(self, district="Kolkata"):
        coords = DISTRICT_COORDINATES.get(district)
        if not coords:
            print(f"⚠️ Warning: District '{district}' not found. Defaulting to Kolkata for historical data.")
            coords = DISTRICT_COORDINATES["Kolkata"]
        
        end_date = datetime.date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=20)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": ["temperature_2m_mean", "rain_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean", "sunshine_duration"],
            "timezone": "auto"
        }
        
        try:
            print(f"Fetching live historical data for {district} from {start_date} to {end_date}...")
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            daily = response.Daily()
            daily_data = {
                "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
                "rain_sum": daily.Variables(1).ValuesAsNumpy(),
                "relative_humidity_2m_mean": daily.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m_mean": daily.Variables(3).ValuesAsNumpy(),
                "sunshine_duration": daily.Variables(4).ValuesAsNumpy()
            }
            
            df = pd.DataFrame(data=daily_data)
            
            aggregated_data = {
                'Past_3W_Temp_Mean': np.nanmean(df['temperature_2m_mean']),
                'Past_3W_Rain_Sum': np.nansum(df['rain_sum']),
                'Past_3W_Humidity_Mean': np.nanmean(df['relative_humidity_2m_mean']),
                'Past_3W_Wind_Mean': np.nanmean(df['wind_speed_10m_mean']),
                'Past_3W_Sunshine_Sum': np.nansum(df['sunshine_duration']),
                'Past_3W_Temp_Std': np.nanstd(df['temperature_2m_mean'])
            }
            print(f"✅ Live historical data aggregated: TempMean {aggregated_data['Past_3W_Temp_Mean']:.1f}C, RainSum {aggregated_data['Past_3W_Rain_Sum']:.1f}mm")
            return aggregated_data
        
        except Exception as e:
            print(f"❌ Historical Weather API error: {e}. Will fall back to CSV data.")
            return None
    
    def get_weather_description(self, rain):
        if rain == 0: return "Clear sky"
        elif rain < 5: return "Light rain"
        elif rain < 15: return "Moderate rain"
        else: return "Heavy rain"
    
    def create_fallback_weather(self):
        print("❌ API FAILED. Using dummy fallback weather.")
        forecasts = []
        base_date = datetime.datetime.now()
        for i in range(14):
            date = base_date + timedelta(days=i) # <-- 2. START DATE FIX
            temp_mean = 26 + np.random.uniform(-2, 2)
            rain = max(0, np.random.normal(1, 2))
            forecasts.append({'date': date.strftime('%d-%m-%Y'), # <-- 1. DATE FORMAT FIX
                              'temp_mean': float(round(temp_mean, 1)),
                              'temp_min': float(round(temp_mean - 3, 1)),
                              'temp_max': float(round(temp_mean + 4, 1)),
                              'rain': float(round(rain, 1)),
                              'humidity': int(75 + np.random.randint(-10, 10)),
                              'wind_speed': float(round(3 + np.random.uniform(0, 2), 1)),
                              'sunshine': int(60000 + np.random.randint(0, 20000)),
                              'description': self.get_weather_description(rain)})
        return forecasts
    
    def create_prediction_features(self, current_price, weather_forecast, day_index, start_price_row, live_historical_data_override):
        weather = weather_forecast[day_index]
        date_obj = datetime.datetime.strptime(weather['date'], '%d-%m-%Y') # <-- 3. DATE PARSING FIX
        features = {}
        
        hist_data_source = live_historical_data_override if live_historical_data_override is not None else start_price_row
        
        features['Current_Week_Min_Price'] = current_price * 0.95
        features['Current_Week_Max_Price'] = current_price * 1.05
        features['Current_Week_Mean_Price'] = current_price
        features['Current_Week_Std_Price'] = current_price * 0.03
        features['Price_Range_Week'] = current_price * 0.1
        
        features['Forecast_Temp_Mean_Next_Week'] = weather['temp_mean']
        features['Forecast_Temp_Min_Next_Week'] = weather['temp_min']
        features['Forecast_Temp_Max_Next_Week'] = weather['temp_max']
        features['Forecast_Rain_Sum_Next_Week'] = weather['rain']
        features['Forecast_Humidity_Mean_Next_Week'] = weather['humidity']
        features['Forecast_Wind_Mean_Next_Week'] = weather['wind_speed']
        features['Forecast_Sunshine_Sum_Next_Week'] = weather['sunshine']
        
        features['Rolling_3W_Mean'] = start_price_row.get('Rolling_3W_Mean', current_price)
        features['Rolling_3W_Std'] = start_price_row.get('Rolling_3W_Std', current_price * 0.04)
        features['Pct_Change_Last_Week'] = start_price_row.get('Pct_Change_Last_Week', 0)
        features['Price_Trend_Slope_3W'] = start_price_row.get('Price_Trend_Slope_3W', 0)
        features['Price_Velocity'] = start_price_row.get('Price_Velocity', 0)
        features['Price_Acceleration'] = start_price_row.get('Price_Acceleration', 0)
        features['Momentum_Strength'] = start_price_row.get('Momentum_Strength', 0.5)
        features['Price_Position_Ratio'] = start_price_row.get('Price_Position_Ratio', 0.5)
        
        features['Next_Week_Month'] = date_obj.month
        features['Is_Monsoon_Next_Week'] = 1 if 6 <= date_obj.month <= 9 else 0
        features['Is_Harvest_Season_Next_Week'] = 1 if date_obj.month in [10, 11] else 0
        features['Is_Festive_Week_Next_Week'] = 0
        
        features['Heat_Stress_Days_Next_Week'] = 1 if weather['temp_max'] > 35 else 0
        features['Cold_Stress_Days_Next_Week'] = 1 if weather['temp_min'] < 15 else 0
        features['Heavy_Rain_Indicator_Next_Week'] = 1 if weather['rain'] > 20 else 0
        features['Drought_Indicator_Next_Week'] = 1 if weather['rain'] < 1 and weather['temp_max'] > 32 else 0
        
        features['Past_3W_Temp_Mean'] = hist_data_source.get('Past_3W_Temp_Mean', 0)
        features['Past_3W_Rain_Sum'] = hist_data_source.get('Past_3W_Rain_Sum', 0)
        features['Past_3W_Humidity_Mean'] = hist_data_source.get('Past_3W_Humidity_Mean', 0)
        features['Past_3W_Wind_Mean'] = hist_data_source.get('Past_3W_Wind_Mean', 0)
        features['Past_3W_Sunshine_Sum'] = hist_data_source.get('Past_3W_Sunshine_Sum', 0)
        features['Past_3W_Temp_Std'] = hist_data_source.get('Past_3W_Temp_Std', 0)
        
        features['Price_Anomaly'] = current_price - features['Rolling_3W_Mean']
        features['Heat_x_Drought'] = features['Forecast_Temp_Max_Next_Week'] * features['Drought_Indicator_Next_Week']
        features['Rain_x_Monsoon'] = features['Forecast_Rain_Sum_Next_Week'] * features['Is_Monsoon_Next_Week']
        
        features['Temp_Anomaly'] = hist_data_source.get('Temp_Anomaly', 0)
        features['Rain_Anomaly'] = hist_data_source.get('Rain_Anomaly', 0)
        features['Volatility_Rain_Interaction'] = hist_data_source.get('Volatility_Rain_Interaction', 0)
        features['Temp_Volatility_Next_Week'] = 0
        features['Humidity_Volatility_Next_Week'] = 0
        
        for feature in self.feature_names:
            if feature not in features:
                features[feature] = 0
        
        return features
    
    def predict_price(self, features):
        try:
            # --- START: FIX FOR UserWarning ---
            input_df = pd.DataFrame([features])
            input_df_ordered = input_df.reindex(columns=self.feature_names)
            input_imputed = self.imputer.transform(input_df_ordered)
            input_scaled = self.scaler.transform(input_imputed)
            # --- END OF FIX ---

            prediction = self.model.predict(input_scaled)[0]
            prediction_float = float(prediction)
            
            return max(100, min(20000, prediction_float))
        
        except Exception as e:
            print(f"Prediction error: {e}")
            missing_keys = [f for f in self.feature_names if f not in features]
            if missing_keys:
                print(f"Missing keys in features dict: {missing_keys[:5]}")
            return features['Current_Week_Mean_Price'] * (1 + np.random.uniform(-0.05, 0.05))


# --- Specialist Model Loader ---
def get_predictor(commodity):
    if commodity in MODEL_CACHE:
        print(f"Returning cached model for {commodity}.")
        return MODEL_CACHE[commodity]
    
    model_filename = f"trained_{commodity.lower().replace(' ', '_')}_model.pkl"
    if not os.path.exists(model_filename):
        print(f"❌ ERROR: Model file not found: {model_filename}")
        return None
    
    try:
        print(f"Loading specialist model from {model_filename}...")
        predictor = CommodityPricePredictor()
        if predictor.load_model(model_filename):
            MODEL_CACHE[commodity] = predictor
            return predictor
        else:
            return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_filename}: {e}")
        return None

# --- Flask Routes ---

@app.route('/')
def home():
    if not data_loaded:
        return "Error: Could not load dropdown data CSV file. Please check server logs.", 500
    return render_template('index.html', commodities=COMMODITY_NAMES)

# --- API: Get Varieties ---
@app.route('/api/get-varieties', methods=['POST'])
def get_varieties():
    if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
    data = request.json
    commodity = data.get('commodity')
    
    filtered_df = DF_FULL
    
    if f"Commodity_{commodity}" in DF_FULL.columns:
        filtered_df = DF_FULL[DF_FULL[f"Commodity_{commodity}"] == 1]
    else:
        return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
    varieties = get_options_from_df(filtered_df, 'Variety_')
    return jsonify({'success': True, 'varieties': varieties})

# --- API: Get Districts ---
@app.route('/api/get-districts', methods=['POST'])
def get_districts():
    if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
    data = request.json
    commodity = data.get('commodity')
    variety = data.get('variety')
    
    filtered_df = DF_FULL
    
    if f"Commodity_{commodity}" in DF_FULL.columns:
        filtered_df = filtered_df[filtered_df[f"Commodity_{commodity}"] == 1]
    else:
        return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
    variety_col = f"Variety_{variety}"
    if variety_col not in DF_FULL.columns:
        return jsonify({'success': False, 'error': f"Invalid variety: {variety}"})
    
    filtered_df = filtered_df[filtered_df[variety_col] == 1]
    
    districts_from_data = get_options_from_df(filtered_df, 'District_')
    valid_districts = [d for d in districts_from_data if d in VALID_DISTRICTS_FROM_COORDS]
    
    return jsonify({'success': True, 'districts': valid_districts})

# --- API: Get Markets ---
@app.route('/api/get-markets', methods=['POST'])
def get_markets():
    if not data_loaded: return jsonify({'success': False, 'error': 'Data not loaded on server.'})
    data = request.json
    commodity = data.get('commodity')
    variety = data.get('variety')
    district = data.get('district')
    
    filtered_df = DF_FULL
    
    # Filter by Commodity
    if f"Commodity_{commodity}" in DF_FULL.columns:
        filtered_df = filtered_df[filtered_df[f"Commodity_{commodity}"] == 1]
    else:
        return jsonify({'success': False, 'error': f"Invalid commodity: {commodity}"})
    
    # Filter by Variety
    variety_col = f"Variety_{variety}"
    if variety_col not in DF_FULL.columns:
        return jsonify({'success': False, 'error': f"Invalid variety: {variety}"})
    filtered_df = filtered_df[filtered_df[variety_col] == 1]

    # Filter by District
    district_col = f"District_{district}"
    if district_col not in DF_FULL.columns:
        return jsonify({'success': False, 'error': f"Invalid district: {district}"})
    filtered_df = filtered_df[filtered_df[district_col] == 1]
    
    # --- START OF FIX ---
    # Get available markets, but ignore "Activity" or other non-market columns
    ignore_list = ['Activity'] 
    markets = get_options_from_df(filtered_df, 'Market_', ignore_list=ignore_list)
    # --- END OF FIX ---
    
    return jsonify({'success': True, 'markets': markets})

# --- Helper function for dropdown APIs ---
# --- START OF FIX ---
# Updated function to accept an ignore_list
def get_options_from_df(df, prefix, ignore_list=None):
    if ignore_list is None:
        ignore_list = []
        
    cols = [col for col in df.columns if col.startswith(prefix)]
    options = []
    for col in cols:
        if df[col].any():
            option_name = col.split('_', 1)[1]
            if option_name not in ignore_list: # Check against the ignore list
                options.append(option_name)
    return sorted(list(set(options)))
# --- END OF FIX ---

# --- /api/predict Route ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        district = data.get('district')
        commodity = data.get('commodity')
        variety = data.get('variety')
        market = data.get('market')
        
        if not district or not variety or not commodity or not market:
            return jsonify({'success': False, 'error': 'Missing input: Commodity, Variety, District, and Market are required.'})
        
        predictor = get_predictor(commodity)
        if predictor is None:
            return jsonify({'success': False, 'error': f"No model available for '{commodity}'. Please train it first."})
        
        live_historical_data = predictor.get_historical_weather(district)
        
        historical_row, (start_price, error) = get_most_recent_price( district, variety, commodity, market)
        if error:
            return jsonify({'success': False, 'error': error})
        
        weather_forecasts = predictor.get_weather_forecast(district)
        if not weather_forecasts:
            return jsonify({'success': False, 'error': 'Could not retrieve weather data.'})
        
        predictions = []
        last_price = start_price
        current_historical_row = historical_row.copy()
        
        for i in range(min(14, len(weather_forecasts))):
            hist_data_for_loop = live_historical_data if i == 0 else None
            
            features = predictor.create_prediction_features(
                last_price,
                weather_forecasts,
                i,
                current_historical_row,
                hist_data_for_loop
            )
            
            predicted_price = predictor.predict_price(features)
            
            predictions.append({
                'date': weather_forecasts[i]['date'],
                'predicted_price': round(predicted_price, 2),
                'weather': weather_forecasts[i],
                'change_percent': round(((predicted_price - start_price) / start_price) * 100, 2),
                'type': 'Predicted'
            })
            
            last_price = predicted_price
            
            new_historical_row = current_historical_row.copy()
            lag_1w = current_historical_row.get('Current_Week_Mean_Price', last_price)
            lag_2w = current_historical_row.get('Lag_1W_Price', last_price)
            
            new_historical_row['Rolling_3W_Mean'] = (lag_1w + lag_2w + last_price) / 3
            new_historical_row['Pct_Change_Last_Week'] = (last_price - lag_1w) / (lag_1w + 1e-6)
            new_historical_row['Current_Week_Mean_Price'] = last_price
            new_historical_row['Lag_1W_Price'] = lag_1w
            new_historical_row['Lag_2W_Price'] = lag_2w
            
            current_historical_row = new_historical_row
        
        return jsonify({
            'success': True,
            'start_price': start_price,
            'district': district,
            'commodity': commodity,
            'variety': variety,
            'market': market,
            'predicted_data': predictions
        })
    
    except Exception as e:
        print(f"Error in /api/predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    if data_loaded:
        print("🚀 Starting Specialist Commodity Price Predictor")
        print("📈 Dropdown data loaded. Models will be loaded on-demand.")
        print("🌦️ Using Open-Meteo for live weather data.")
        print("🌐 Open: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Application failed to start. Please fix the CSV file path.")