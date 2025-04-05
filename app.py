print("🚀 Streamlit app is running")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Career Prediction App", layout="wide")

st.title("🎓 Career Prediction Based on Interests & Personality")

# File uploader
uploaded_file = st.file_uploader("Upload your training CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop unnecessary or irrelevant columns
    if "Timestamp" in df.columns:
        df.drop(columns=["Timestamp"], inplace=True)

    irrelevant_cols = [
        'Favorite Color',
        'Daily Water Intake',
        'Birth Month',
        'Preferred Music Genre',
        'Number of Siblings'
    ]
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)

    # Target column
    target_col = "What would you like to become when you grow up"
    df.dropna(subset=[target_col], inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Display class distribution before balancing
    st.subheader("🎯 Class Distribution Before Balancing")
    st.bar_chart(y.value_counts())

    # Manual Oversampling
    def manual_oversample(X, y):
        df_combined = X.copy()
        df_combined['target'] = y
        max_size = df_combined['target'].value_counts().max()
        lst = [df_combined]
        for _, group in df_combined.groupby('target'):
            lst.append(group.sample(max_size - len(group), replace=True))
        df_balanced = pd.concat(lst)
        return df_balanced.drop('target', axis=1), df_balanced['target']

    X_resampled, y_resampled = manual_oversample(X, y)

    st.subheader("✅ Class Distribution After Balancing")
    st.bar_chart(pd.Series(y_resampled).value_counts())

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy & report
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained with accuracy: **{acc:.2f}**")

    # Feature importance
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
    st.bar_chart(data=feat_imp.set_index("Feature"))

    # Prediction input
    st.subheader("🧠 Predict Your Career Path")
    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            user_input[col] = st.number_input(f"{col}", min_value=0, max_value=100, step=1)

    if st.button("Predict Career"):
        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                le = label_encoders[col]
                input_df[col] = input_df[col].map(lambda s: s if s in le.classes_ else 'Unknown')
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                input_df[col] = le.transform(input_df[col])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        pred = model.predict(input_df)[0]
        result = label_encoders[target_col].inverse_transform([pred])[0]
        st.success(f"🎯 Based on your responses, a suitable career path is: **{result}**")


