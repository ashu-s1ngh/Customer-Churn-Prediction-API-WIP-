# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib

# --- 2. Load Data ---
# Load the raw dataset from a CSV file
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display a summary of the DataFrame (column names, non-null counts, data types)
df.info()

# --- 3. Data Cleaning & Type Conversion ---

# 'TotalCharges' is incorrectly loaded as an 'object' (string) type.
# We must convert it to a numeric type to use it in the model.
# errors='coerce' will turn any non-numeric strings (like ' ') into NaN (Not a Number)
df["total_charges_cleaned"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Verify the data types 'before' and 'after' the conversion
print(f"Original 'TotalCharges' dtype: {df['TotalCharges'].dtype}")
print(f"New 'total_charges_cleaned' dtype: {df['total_charges_cleaned'].dtype}")

# Check how many rows were turned into NaN (i.e., were blank or non-numeric)
missing_count = df['total_charges_cleaned'].isnull().sum()
print(f"Number of rows with missing TotalCharges: {missing_count}")

# Optional: Display the actual rows that had missing values
# We can see these are customers with 0 tenure.
# print(df[df['total_charges_cleaned'].isnull()])

# Fill the missing values.
# Based on the data, these are 0-tenure customers, so '0' is the logical fill value.
df['total_charges_cleaned'] = df['total_charges_cleaned'].fillna(0)

# Verify that all missing values have been filled
filtered_missing_counts = df['total_charges_cleaned'].isnull().sum()
print(f"Missing charges after fillna: {filtered_missing_counts}")

# Drop the original, messy 'TotalCharges' column and the non-predictive 'customerID'
df = df.drop('TotalCharges', axis=1)
df = df.drop('customerID', axis=1)

# Display info again to confirm our changes
df.info()

# --- 4. Target Variable Encoding ---

# Convert the 'Churn' column from 'Yes'/'No' strings to 1/0 integers
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Check the distribution of the target variable (it's imbalanced)
print(df['Churn'].value_counts())

# --- 5. Feature Engineering ---

# Create a list of all binary 'service' columns
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Create a temporary DataFrame where 'Yes'=1, 'No'=0, and 'No internet service'=0
df_services = df[services].replace({'Yes': 1, 'No': 0, 'No internet service': 0})

# Create a new feature 'ServiceCount' by summing up the number of services
df['ServiceCount'] = df_services.sum(axis=1)

# Create a new feature 'Monthly_Charges_per_Service'
# We add 1 to the denominator to avoid divide-by-zero errors for customers with 0 services
df['Monthly_Charges_per_Service'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)

# Optional: Display the head of the new features
# print(df[['MonthlyCharges', 'ServiceCount', 'Monthly_Charges_per_Service']].head)

# --- 6. Define Features (X) and Target (y) ---

# Define the target variable name
TARGET = 'Churn'

# Create the target variable Series (y)
y = df[TARGET]
# Create the features DataFrame (X) by dropping the target
x = df.drop(TARGET, axis=1)

# Optional: Check the first few rows
# print(x.head)
# print(y.head)

# --- 7. Train-Test Split ---

# Split the data into training and testing sets (80% train, 20% test)
# 'random_state=42' ensures the split is reproducible
# 'stratify=y' is CRITICAL for imbalanced datasets. It ensures that both the
# training and test sets have the same percentage of churners as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Print shapes and churn rates to verify the split and stratification
print(f"X_train and X_test has {X_train.shape[0]} and {X_test.shape[0]} rows respectively")
print("Churn Rate Verification (Stratify):")
print(f"Original dataset: {y.mean():.2%}")
print(f"Training set:   {y_train.mean():.2%}")
print(f"Testing set:    {y_test.mean():.2%}")

# --- 8. Define Preprocessing Pipeline ---

# Define the lists of numerical and categorical feature names
numerical_features = [
    'tenure',
    'MonthlyCharges',
    'total_charges_cleaned',
    'ServiceCount',
    'Monthly_Charges_per_Service'
]

categorical_features = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

# Sanity check: Ensure all columns from X_train are accounted for
print(f"We found {len(numerical_features)} numerical features.")
print(f"We found {len(categorical_features)} categorical features.")
print(f"Total features: {len(numerical_features) + len(categorical_features)}")
print(f"Original X_train columns: {len(X_train.columns)}")

assert len(numerical_features) + len(categorical_features) == len(X_train.columns), "Error: The feature lists do not match the number of columns in X_train."
print("\nSuccess! All columns are accounted for.")


# Create the preprocessing pipeline for numerical features
# We will scale them using StandardScaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create the preprocessing pipeline for categorical features
# We will one-hot encode them
# 'handle_unknown='ignore'' prevents errors if the API receives a category
# that wasn't in the training data (it will just be encoded as all zeros).
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the master preprocessor using ColumnTransformer
# This applies the correct transformer to the correct set of columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# --- 9. Fit and Transform Data ---

# Fit the preprocessor *ONLY* on the training data
# This learns the scaling parameters (mean, std) and one-hot categories
print("Fitting the preprocessor on X_train...")
preprocessor.fit(X_train)
print("Fitting complete.")

# Apply the fitted preprocessor to transform both train and test sets
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Print the shapes to see the effect of one-hot encoding (many new columns)
print(f"X_train_original shape: {X_train.shape}")
print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_original shape: {X_test.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

# --- 10. Train Model ---

# Initialize the RandomForestClassifier
# 'n_jobs=-1' uses all available CPU cores for faster training
# 'class_weight='balanced'' is CRITICAL for imbalanced datasets.
# It automatically adjusts weights to give more importance to the minority class (Churn=1).
model = RandomForestClassifier(
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)

# Train the model on the *processed* training data
print("Starting model training...")
model.fit(X_train_processed, y_train)
print("Training complete.")

# --- 11. Evaluate Model ---

# Get probability predictions for the positive class (Churn=1)
# This is needed for the ROC AUC score
y_pred_prob = model.predict_proba(X_test_processed)[:, 1]

# Get binary (0 or 1) predictions
y_pred_binary = model.predict(X_test_processed)

# Calculate key performance metrics
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)

# Print the metrics
print(f"\n--- Model Evaluation ---")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=['Not Churn (0)', 'Churn (1)']))

# --- 12. Save Artifacts ---

# Save the *fitted* preprocessor pipeline to a file
# This is the single most important artifact for your API,
# as it ensures new data is transformed in the *exact* same way.
joblib.dump(preprocessor, 'preprocessor_pipeline.joblib')

# Save the *trained* model to a file
joblib.dump(model, 'churn_model.joblib')

print("\n--- Artifacts Saved ---")
print("Both the preprocessor ('preprocessor_pipeline.joblib') and")
print("the model ('churn_model.joblib') are saved to disk.")