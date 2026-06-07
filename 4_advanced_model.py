import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data preparation
df = pd.read_csv(
    "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv"
)
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# training random forest for better accuracy
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# predictions
y_pred_rf = rf_model.predict(X_test)

# checking classification report
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf))

# plotting confusion matrix to see false positives/negatives
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"],
)
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

# final stats
print("\n--- Confusion Matrix Metrics ---")
print("True Negatives:", cm[0][0])
print("False Positives (False Alarms):", cm[0][1])
print("False Negatives (Missed Frauds):", cm[1][0])
print("True Positives (Detected Frauds):", cm[1][1])
