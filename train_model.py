import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and clean your original data
df = pd.read_csv("../train_u6lujuX_CVtuZ9i.csv")  # Adjust path if needed

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.fillna(method="ffill").fillna(0)

# Move the label column to y BEFORE encoding
y = df["Loan_Status"]

# Encode the rest of the dataframe (excluding the label)
df_features = df.drop("Loan_Status", axis=1)

# Preprocessing
df_features["Dependents"] = df_features["Dependents"].replace("3+", 3).astype(float)
df_features = pd.get_dummies(df_features, drop_first=True)
df_features = df_features.fillna(method="ffill").fillna(0)

# Continue training
X = df_features

# Save the column names used during training
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model in SAME environment
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save to correct folder
joblib.dump(model, "loan_model.pkl")
print("Model re-trained and saved from the correct environment.")
