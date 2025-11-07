import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib

df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.info()

df["total_charges_cleaned"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
print(df['TotalCharges'].dtype)
print(df['total_charges_cleaned'].dtype)

df['total_charges_cleaned']

missing_count = df['total_charges_cleaned'].isnull().sum()
print(missing_count)

df[df['total_charges_cleaned'].isnull()]

df['total_charges_cleaned'] = df['total_charges_cleaned'].fillna(0)
filtered_missing_counts = df['total_charges_cleaned'].isnull().sum()
print(filtered_missing_counts)

df = df.drop('TotalCharges', axis = 1)
df = df.drop('customerID', axis = 1)
df.info()

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df['Churn'].value_counts())

services = ['OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport', 'StreamingTV','StreamingMovies']
df_services = df[services].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df['ServiceCount'] = df_services.sum(axis = 1)
df['Monthly_Charges_per_Service'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
print(df[['MonthlyCharges', 'ServiceCount', 'Monthly_Charges_per_Service']].head)

TARGET = 'Churn'
y = df[TARGET]
x = df.drop(TARGET, axis = 1)
print(x.head)
print(y.head)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify= y)
print("X_train and X_test has", X_train.shape[0], X_test.shape[0], "rows")
print("churns")
print(f"original {y.mean():.2%}")
print(f"y_train {y_train.mean():.2%}")
print(f"y_test {y_test.mean():.2%}")

# X_train.info()
# X_test.info()
# y_train.info()
y_test.info()

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
print(f"We found {len(numerical_features)} numerical features.")
print(f"We found {len(categorical_features)} categorical features.")
print(f"Total features: {len(numerical_features) + len(categorical_features)}")
print(f"Original X_train columns: {len(X_train.columns)}")

assert len(numerical_features) + len(categorical_features) == len(X_train.columns), "Error: The feature lists do not match the number of columns in X_train."
print("\nSuccess! All columns are accounted for.")

numeric_transformer = Pipeline(steps = [
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])
print("Numeric Translator Pipeline")
print(numeric_transformer)
print("Categorical Translator Pipeline")
print(categorical_transformer)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder = 'drop'
)

print("Master Preprocessor")
print(preprocessor)

preprocessor.fit(X_train)
print("Fitting complete")

X_train_processed = preprocessor.transform(X_train)
print(f"X_train_original shape: {X_train.shape}")
print("X_train_processed shape")
print(X_train_processed.shape)
X_test_processed = preprocessor.transform(X_test)
print(f"X_test_original shape: {X_test.shape}")
print("X_test_processed shape")
print(X_test_processed.shape)

model = RandomForestClassifier(
    n_jobs=-1,
    class_weight='balanced',
    random_state = 42
    )
print("Starting model training")
model.fit(X_train_processed, y_train)
print("Training complete")

y_pred_prob = model.predict_proba(X_test_processed)[:, 1]
y_pred_binary = model.predict(X_test_processed)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred_binary, target_names=['Not Churm (0)', 'Churn (1)']))

joblib.dump(preprocessor, 'preprocessor_pipeline.joblib')
joblib.dump(model, 'churn_model.joblib')
print("Both the preprocessor and model are saved!!!!")