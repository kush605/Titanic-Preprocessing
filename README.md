# Titanic-Preprocessing

Titanic Dataset - Data Cleaning & Preprocessing 🛠️

This project demonstrates how to clean and preprocess the Titanic dataset for Machine Learning.
It covers essential steps such as handling missing values, encoding categorical variables, removing outliers, and scaling features.

 Objective:

Prepare raw Titanic data into a cleaned version ready for ML models.

Tools & Libraries:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

Steps Performed:

1. Import Dataset
df = pd.read_csv("Titanic-Dataset.csv")

2. Data Cleaning
Dropped irrelevant columns: PassengerId, Name, Ticket, Cabin
Filled missing values:
Age → median
Embarked → mode

3. Encoding Categorical Features
Converted Sex into numeric (Label Encoding)
One-hot encoded Embarked

4. Outlier Detection & Removal
Visualized outliers with boxplots
Removed extreme outliers in Fare using IQR method

5. Feature Scaling
Standardized Age and Fare with StandardScaler

6. Save Final Dataset
df.to_csv("Titanic-Dataset-Cleaned.csv", index=False)


The cleaned dataset is now ready for ML tasks 🚀.

📂 Project Structure
├── Titanic-Dataset.csv            # Raw dataset
├── Titanic-Dataset-Cleaned.csv    # Cleaned dataset (output)
├── titanic_clean.py               # Preprocessing script
└── README.md                      # Project documentation

Next Steps:

Use Titanic-Dataset-Cleaned.csv for ML models (Logistic Regression, Random Forest, etc.)

Experiment with feature engineering and model evaluation.

👨‍💻 Author

Project completed by Kush saini as part of Elevate Labs AI/ML Internship Task.
