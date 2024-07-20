import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv('/content/ml dataset1.csv')
X = df[['Traffic Volume']]

isolation_forest = IsolationForest(contamination=0.05, random_state=42)

isolation_forest.fit(X)
df['anomaly_score'] = isolation_forest.decision_function(X)
df['anomaly'] = isolation_forest.predict(X)

plt.figure(figsize=(10, 6))
plt.hist(df['anomaly_score'], bins='auto')
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
anomalies = df[df['anomaly'] == -1]
anomalies.to_csv('/content/anomalies.csv', index=False)
