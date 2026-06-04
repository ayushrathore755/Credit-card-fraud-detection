import os
import pandas as pd

# Load the dataset
# Note: Update this path according to your environment setup
df = pd.read_csv('/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv')

# Display dataset shape and preview
print("Dataset Shape:", df.shape)
print(df.head())

# Check class distribution and percentage
print("\nClass Distribution (0: Normal, 1: Fraud):")
print(df['Class'].value_counts())
print("\nClass Percentage:")
print(df['Class'].value_counts(normalize=True) * 100)
