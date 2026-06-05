import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv')

# Plotting how many transactions are fraud vs normal
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Normal Transactions')
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# Comparing transaction amounts for both classes side by side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50, color='blue', alpha=0.7)
plt.title('Normal Transaction Amounts')
plt.xlabel('Amount (USD)')

plt.subplot(1, 2, 2)
df[df['Class'] == 1]['Amount'].hist(bins=50, color='red', alpha=0.7)
plt.title('Fraudulent Transaction Amounts')
plt.xlabel('Amount (USD)')

plt.tight_layout()
plt.savefig('amount_distribution.png')
plt.show()

# Fraud transactions generally have lower amounts than expected
print("Average Normal Transaction Amount    :", round(df[df['Class'] == 0]['Amount'].mean(), 2))
print("Average Fraudulent Transaction Amount:", round(df[df['Class'] == 1]['Amount'].mean(), 2))
