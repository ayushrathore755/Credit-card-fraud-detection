from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# load data
df = pd.read_csv(
    "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv"
)

# split features and target variable
X = df.drop("Class", axis=1)
y = df["Class"]

# scale features using standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80-20 train test split (using stratify because data is highly imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

# baseline model - logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# evaluate baseline performance
y_pred_lr = lr_model.predict(X_test)
print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_lr))
