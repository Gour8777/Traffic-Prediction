import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None

def preprocess_data(df, encoder=None):
    try:
        # Extract hour from the 'Hour' column and handle '-' format
        df['Hour'] = df['Hour'].apply(lambda x: int(x.split(':')[0]) + 0.5 if '-' in x else int(x.split(':')[0]))

        # Extract day and month from the 'Date' column
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month

        # Combine 'From' and 'To' columns to encode together
        combined_categories = pd.concat([df['From'], df['To']])

        if encoder is None:
            encoder = LabelEncoder()
            encoder.fit(combined_categories)

        df['From_encoded'] = encoder.transform(df['From'])
        df['To_encoded'] = encoder.transform(df['To'])

        return df, encoder
    except KeyError as e:
        print(f"Key error: {e}. Please check if all necessary columns are present in the dataframe.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None, None

def train_model(df):
    try:
        X = df[['From_encoded', 'To_encoded', 'Week', 'Hour', 'Day', 'Month']]
        y = df['Traffic Volume'].apply(lambda x: 'high' if x > 30 else 'low')

        if y.isnull().any():
            raise ValueError("Target variable 'y' contains null values.")
        if not set(y.unique()).issubset({'high', 'low'}):
            raise ValueError(f"Target variable 'y' contains unexpected values: {set(y.unique()) - {'high', 'low'}}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

        return clf, clf.classes_
    except KeyError as e:
        print(f"Key error: {e}. Please check if all necessary columns are present in the dataframe.")
        return None, None
    except ValueError as e:
        print(f"Value error: {e}. Please check the target variable 'Traffic Volume' for unexpected values or nulls.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return None, None

def predict_traffic_condition(clf, encoder, df_classes, user_from, user_to, week, hour, date):
    try:
        if user_from not in encoder.classes_ or user_to not in encoder.classes_:
          print("Invalid input: Location not found.")
          return

        user_from_encoded = encoder.transform([user_from])[0]
        user_to_encoded = encoder.transform([user_to])[0]

        user_input = pd.DataFrame({'From_encoded': [user_from_encoded], 'To_encoded': [user_to_encoded], 'Week': [week], 'Hour': [hour], 'Date': [date]})

        # Extract day and month from user-input date
        user_input['Date'] = pd.to_datetime(user_input['Date'], format='%d-%m-%Y')
        user_input['Day'] = user_input['Date'].dt.day
        user_input['Month'] = user_input['Date'].dt.month

        predicted_class = clf.predict(user_input[['From_encoded', 'To_encoded', 'Week', 'Hour', 'Day', 'Month']])[0]
        predicted_condition = 'high' if predicted_class == df_classes[1] else 'low'

        return predicted_condition
    except Exception as e:
        return f"Prediction error: {e}. Please check your inputs and try again."

if __name__ == "__main__":
    try:
        file_path = '/content/ml dataset1.csv'
        df = load_data(file_path)

        if df is not None:
            df, encoder = preprocess_data(df)

            if df is not None and encoder is not None:
                trained_model, df_classes = train_model(df)

                if trained_model is not None and df_classes is not None:
                    user_from = input("Enter 'From' location: ")
                    user_to = input("Enter 'To' location: ")
                    week = int(input("Enter week number (e.g., 2): "))
                    hour = float(input("Enter hour (e.g., 12.0 for 12 PM, 0.5 for 12:30 AM): "))
                    date_str = input("Enter date (format DD-MM-YYYY): ")

                    predicted_condition = predict_traffic_condition(trained_model, encoder, df_classes, user_from, user_to, week, hour, date_str)
                    print(f"Predicted Traffic Condition from {user_from} to {user_to} at week {week}, hour {hour}, and date {date_str}: {predicted_condition}")
                else:
                    print("Failed to train model.")
            else:
                print("Failed to preprocess data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
