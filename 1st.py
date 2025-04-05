#1st run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"D:\anaconda\train_data1.csv")  # Assuming the file is in the same directory

# Drop unnecessary columns
if "Timestamp" in df.columns:
    df.drop(columns=["Timestamp"], inplace=True)

# Set the target column
target_col = "What would you like to become when you grow up"

# Drop rows with missing target
df.dropna(subset=[target_col], inplace=True)

# Fill missing values appropriately
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# Label Encoding for categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Define features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- Predict Career for New Input ---

# Example input from a new user
sample_input_dict = {
    'Your Gender': 'Male',
    'Age': 16,
    'Your favorite subject in school': 'Math',
    'What do you like doing in your free time': 'Reading',
    'Do you enjoy working in teams or alone?': 'Teams',
    'Are you interested in technology?': 'Yes',
    'Do you prefer practical tasks or theoretical learning?': 'Practical',
    'What motivates you the most?': 'Innovation',
    # ⚠️ Add all remaining features used in training if any
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_input_dict])

# Encode sample input using training encoders
for col in sample_df.columns:
    if sample_df[col].dtype == 'object' and col in label_encoders:
        le = label_encoders[col]
        sample_df[col] = sample_df[col].map(lambda s: s if s in le.classes_ else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        sample_df[col] = le.transform(sample_df[col])

# Match feature columns
sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

# Predict career
predicted_label = model.predict(sample_df)[0]
career_prediction = label_encoders[target_col].inverse_transform([predicted_label])[0]

print("\n🎯 Predicted Career Path:", career_prediction)
