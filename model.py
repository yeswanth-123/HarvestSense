#! XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# --- 1. Define the list of files and their "friendly" names ---
COMMODITY_FILES = {
    'Green Chilli': 'Green_Chiili_Without_Dropping.csv',
    'Onion': 'Onion_Without_Dropping.csv',
    'Potato': 'Potato_Without_Dropping.csv', # Handle the space
    'Tomato': 'Tomato_Without_Dropping.csv',
    'Rice': 'Rice_Without_Dropping.csv'
}

# --- 2. Define ALL features to be used for training ---
# --- THIS IS THE CRITICAL UPDATE ---
# We are now including all your "Past_3W_..." features from the CSVs
BASE_FEATURE_COLUMNS = [
    # Price
    'Current_Week_Min_Price', 'Current_Week_Max_Price', 'Current_Week_Mean_Price',
    'Current_Week_Std_Price', 'Price_Range_Week',
    # Weather Forecast
    'Forecast_Temp_Mean_Next_Week', 'Forecast_Temp_Min_Next_Week', 'Forecast_Temp_Max_Next_Week',
    'Forecast_Rain_Sum_Next_Week', 'Forecast_Humidity_Mean_Next_Week',
    'Forecast_Wind_Mean_Next_Week', 'Forecast_Sunshine_Sum_Next_Week',
    # Technical
    'Rolling_3W_Mean', 'Rolling_3W_Std', 'Pct_Change_Last_Week', 
    'Price_Trend_Slope_3W', 'Price_Velocity', 'Price_Acceleration', 
    'Momentum_Strength', 'Price_Position_Ratio',
    # Seasonal
    'Next_Week_Month', 'Is_Monsoon_Next_Week', 'Is_Harvest_Season_Next_Week',
    'Is_Festive_Week_Next_Week',
    # Stress
    'Heat_Stress_Days_Next_Week', 'Cold_Stress_Days_Next_Week',
    'Heavy_Rain_Indicator_Next_Week', 'Drought_Indicator_Next_Week',
    
    # --- NEWLY INCLUDED HISTORICAL FEATURES ---
    'Past_3W_Temp_Mean', 'Past_3W_Temp_Std', 'Past_3W_Rain_Sum', 
    'Past_3W_Humidity_Mean', 'Past_3W_Wind_Mean', 'Past_3W_Sunshine_Sum',
    # -------------------------------------------
    
    # Engineered features (from your CSVs or we will create them)
    'Temp_Anomaly', 'Rain_Anomaly', 'Volatility_Rain_Interaction',
    'Temp_Volatility_Next_Week', 'Humidity_Volatility_Next_Week',
    'Price_Anomaly', 'Heat_x_Drought', 'Rain_x_Monsoon'
]
TARGET_COLUMN = 'Target_Next_Week_Mean_Price'

print(f"--- Starting Specialist Model Training Factory ---")
print(f"Found {len(COMMODITY_FILES)} commodities to train.")

# --- 3. Master Loop: Train one model per file ---
for commodity_name, csv_filename in COMMODITY_FILES.items():
    print(f"\n==============================================")
    print(f"           Training Model for: {commodity_name}")
    print(f"           Loading file: {csv_filename}")
    print(f"==============================================")

    # --- 4. Load Data ---
    if not os.path.exists(csv_filename):
        print(f"SKIPPING: File not found: {csv_filename}")
        continue
        
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"SKIPPING: Error loading {csv_filename}. Error: {e}")
        continue

    # --- 5. Feature Engineering ---
    try:
        if 'Price_Anomaly' not in df.columns:
            df['Price_Anomaly'] = df['Current_Week_Mean_Price'] - df['Rolling_3W_Mean']
        if 'Heat_x_Drought' not in df.columns:
            df['Heat_x_Drought'] = df['Forecast_Temp_Max_Next_Week'] * df['Drought_Indicator_Next_Week']
        if 'Rain_x_Monsoon' not in df.columns:
            df['Rain_x_Monsoon'] = df['Forecast_Rain_Sum_Next_Week'] * df['Is_Monsoon_Next_Week']
    except Exception as e:
        print(f"Warning: Could not create engineered features for {commodity_name}. {e}")

    # --- 6. Prepare Data ---
    available_features = [col for col in BASE_FEATURE_COLUMNS if col in df.columns]
    
    if TARGET_COLUMN not in df.columns:
        print(f"SKIPPING: Target column '{TARGET_COLUMN}' not found in {csv_filename}.")
        continue
    
    missing_hist_features = [f for f in ['Past_3W_Temp_Mean', 'Past_3W_Rain_Sum'] if f not in available_features]
    if missing_hist_features:
        print(f"Warning: Missing key historical features: {missing_hist_features}. Accuracy may be affected.")

    X = df[available_features].copy()
    y = df[TARGET_COLUMN]

    X = X.fillna(0) 
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # --- 7. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False
    )

    if len(X_train) < 50 or len(X_test) < 10:
        print(f"SKIPPING: Not enough data for {commodity_name} (Train: {len(X_train)}, Test: {len(X_test)}).")
        continue
        
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # --- 8. Train Model ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50 # Prevents overfitting
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- 9. Evaluate ---
    y_pred = xgb_model.predict(X_test)
    
    y_test_safe = y_test.copy()
    y_test_safe[y_test_safe == 0] = 1e-6

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test_safe, y_pred) * 100
    
    tolerance_percent = 0.10
    correct_within_tolerance = ((y_pred >= y_test_safe * (1 - tolerance_percent)) & 
                                (y_pred <= y_test_safe * (1 + tolerance_percent))).sum()
    accuracy_within_10_percent = (correct_within_tolerance / len(y_test_safe)) * 100

    print(f"\n--- Performance for {commodity_name} ---")
    print(f"  MAE: ₹{mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Accuracy (within ±10%): {accuracy_within_10_percent:.2f}%")

    # --- 10. Feature Importance ---
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Top 5 Most Important Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # --- 11. Save Model ---
    model_data = {
        'model': xgb_model,
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': available_features,
    }
    
    model_filename = f"trained_{commodity_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model_data, model_filename)
    print(f"\n✅ Model saved as '{model_filename}'")

print("\n--- All Specialist Model Training Complete ---")