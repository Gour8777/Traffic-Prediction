import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

model_rf = RandomForestRegressor(random_state=60)

model_rf.fit(X, y)
def predict_traffic_volume():
    segment_id = int(input("Enter Segment ID: "))
    week = int(input("Enter Week number: "))
    hour_str = input("Enter Hour range (e.g., '12:00-1:00 AM'): ")
    hour = hour_range_to_numeric(hour_str)
    if(segment_id not in df['SegmentID'].values):
        print("Invalid Segment ID")
        return


    new_data_rf = pd.DataFrame({'SegmentID': [segment_id], 'Week': [week], 'Hour': [hour]})


    predicted_volume_rf = model_rf.predict(new_data_rf)

    print(f'Predicted Traffic Volume: {predicted_volume_rf[0]:.2f}')


    y_pred_rf = model_rf.predict(X)
    mse_rf = mean_squared_error(y, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)

    print(f'Random Forest - Mean Squared Error (MSE): {mse_rf:.2f}')
    print(f'Random Forest - Root Mean Squared Error (RMSE): {rmse_rf:.2f}')

predict_traffic_volume()
