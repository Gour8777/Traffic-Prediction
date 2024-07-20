import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

df = pd.read_csv('/content/ml dataset1.csv')

X = df[['Traffic Volume']]

svm = OneClassSVM(nu=0.05)

svm.fit(X)
df['anomaly'] = svm.predict(X)

plt.figure(figsize=(10, 6))
plt.hist(df['anomaly'], bins='auto')
plt.title('Anomaly Detection using One-Class SVM')
plt.xlabel('Anomaly Label (-1: Anomaly, 1: Normal)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

anomalies = df[df['anomaly'] == -1]
print("Detected Anomalies:")
anomalies.to_csv('/content/anomalies_svm.csv', index=False)
