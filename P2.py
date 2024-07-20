import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

def hour_range_to_numeric(hour_str):
    start, end = hour_str.split('-')
    start_hour = int(start.split(':')[0])
    end_hour = int(end.split(':')[0])
    if end_hour == 1:
        end_hour += 12
    return (start_hour + end_hour) / 2

df = pd.read_csv('/content/ml dataset1.csv')

df['Hour'] = df['Hour'].apply(hour_range_to_numeric)

X = df[['SegmentID', 'Week', 'Hour']]
y = df['Traffic Volume']
model_rf = RandomForestRegressor(random_state=42)
model_xgb = XGBRegressor(random_state=42)
ensemble_model = VotingRegressor(estimators=[
    ('Random Forest', model_rf),
    ('XGBoost', model_xgb)
])
ensemble_model.fit(X, y)
def predict_traffic_volume():
    segment_id = int(input("Enter Segment ID: "))
    week = int(input("Enter Week number: "))
    hour_str = input("Enter Hour range (e.g., '12:00-1:00 AM'): ")
    hour = hour_range_to_numeric(hour_str)
    if(segment_id not in df['SegmentID'].values):
        print("Invalid Segment ID")
        return

    new_data = pd.DataFrame({'SegmentID': [segment_id], 'Week': [week], 'Hour': [hour]})

    predicted_volume = ensemble_model.predict(new_data)

    print(f'Predicted Traffic Volume (Ensemble): {predicted_volume[0]:.2f}')


    y_pred = ensemble_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f'Ensemble - Mean Squared Error (MSE): {mse:.2f}')
    print(f'Ensemble - Root Mean Squared Error (RMSE): {rmse:.2f}')
predict_traffic_volume()
