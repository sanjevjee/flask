import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("breast_cancer_dataset.csv")

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
df = df.dropna(subset=["diagnosis"])  # Ensure no missing target values

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Separate numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=50,
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
pipeline.fit(X_train, y_train)

# Save trained model
joblib.dump(pipeline, "model.pkl")
print("✅ Model trained and saved to model.pkl")
