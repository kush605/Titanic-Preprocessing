import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================
# Load Dataset
# =============================
df = pd.read_csv("Titanic-Dataset.csv")

# =============================
# Data Cleaning
# =============================
# Drop irrelevant columns
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# =============================
# Encoding Categorical Features
# =============================
# Encode Sex
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# =============================
# Outlier Detection & Removal
# =============================
# Function to remove outliers based on IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Visualize outliers before removal
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers Before Removal")
plt.show()

# Remove outliers from Fare
df = remove_outliers_iqr(df, "Fare")

# Visualize outliers after removal
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers After Removal")
plt.show()

# =============================
# Feature Scaling
# =============================
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# =============================
# Final Cleaned Data
# =============================
print("Final Cleaned Data:")
print(df.head())

# Save preprocessed dataset
df.to_csv("Titanic-Dataset-Cleaned.csv", index=False)
print("Preprocessed dataset saved as 'Titanic-Dataset-Cleaned.csv'")
