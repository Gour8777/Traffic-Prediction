import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, encoder=None):

    df['Hour'] = df['Hour'].apply(lambda x: int(x.split(':')[0]) + 0.5 if '-' in x else int(x.split(':')[0]))


    combined_categories = pd.concat([df['From'], df['To']])


    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(combined_categories)

    df['From_encoded'] = encoder.transform(df['From'])
    df['To_encoded'] = encoder.transform(df['To'])

    return df, encoder


def train_model(df):
    try:
        X = df[['From_encoded', 'To_encoded', 'Week', 'Hour']]
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

        return clf, encoder
    except KeyError as e:
        print(f"Key error: {e}. Please check if all necessary columns are present in the dataframe.")
    except ValueError as e:
        print(f"Value error: {e}. Please check the target variable 'Traffic Volume' for unexpected values or nulls.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def predict_traffic_condition(clf, encoder, user_from, user_to, week, hour):
    if user_from not in df['From'].values or user_to not in df['To'].values:
        return "Invalid input"
    else:
        user_from_encoded = encoder.transform([user_from])[0]
        user_to_encoded = encoder.transform([user_to])[0]
        user_input = pd.DataFrame({'From_encoded': [user_from_encoded], 'To_encoded': [user_to_encoded], 'Week': [week], 'Hour': [hour]})
        predicted_condition = clf.predict(user_input)[0]
        return predicted_condition


if __name__ == "__main__":

    file_path = '/content/ml dataset1.csv'
    df = load_data(file_path)
    df, encoder = preprocess_data(df)


    trained_model, encoder = train_model(df)

    if trained_model is not None:

        user_from = input("Enter 'From' location: ")
        user_to = input("Enter 'To' location: ")
        week = int(input("Enter week number (e.g., 2): "))
        hour = float(input("Enter hour (e.g., 12.0 for 12 PM, 0.5 for 12:30 AM): "))


        predicted_condition = predict_traffic_condition(trained_model, encoder, user_from, user_to, week, hour)
        print(f"Predicted Traffic Condition from {user_from} to {user_to} at week {week} and hour {hour}: {predicted_condition}")
