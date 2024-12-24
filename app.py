
import streamlit as st
import pandas as pd
from explore import explore_data
from main import clean_dataset
from model import model_comparison
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Dataset Cleaner", "Explore Data", "Original vs Cleaned", "Original vs Synthetic"])

if page == "Dataset Cleaner":
    st.title("Dataset Cleaner")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Original Dataset")
        st.dataframe(data.head(10))

        sensitive_columns, cleaned_data = clean_dataset(data)
        st.subheader("Sensitive Columns Detected")
        st.write(sensitive_columns)

        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_data.head(10))

        cleaned_data.to_csv("cleaned_dataset.csv", index=False)
        st.success("Cleaned dataset saved!")

elif page == "Explore Data":
    st.title("Explore Data")

    new_dataset = pd.read_csv("new_dataset.csv")
    cleaned_dataset = pd.read_csv("cleaned_dataset.csv")

    explore_data(new_dataset, cleaned_dataset)

elif page == "Original vs Cleaned":
    st.title("Original vs Cleaned Dataset")
    new_dataset = pd.read_csv("new_dataset.csv")
    cleaned_dataset = pd.read_csv("cleaned_dataset.csv")
    model_comparison(new_dataset, cleaned_dataset)

elif page == "Original vs Synthetic":
    st.title("Original vs Synthetic Dataset Models")

    # Load datasets
    new_dataset = pd.read_csv("new_dataset.csv")
    anonymized_dataset = pd.read_csv("anonymized_dataset.csv")

    # st.write("### New Dataset (First 10 Rows)")
    # st.dataframe(new_dataset.head(10))

    # st.write("### Anonymized Dataset (First 10 Rows)")
    # st.dataframe(anonymized_dataset.head(10))

    # Define target and features
    target_column = 'Salary'
    features = ['Experience', 'Education', 'Age']

    # Preprocess datasets
    def preprocess_dataset(dataset):
        le = LabelEncoder()
        for col in dataset.select_dtypes(include=['object']).columns:
            dataset[col] = le.fit_transform(dataset[col].astype(str))
        dataset = dataset.apply(pd.to_numeric, errors='coerce').dropna()
        X = dataset[features]
        y = dataset[target_column]
        return X, y

    X_new, y_new = preprocess_dataset(new_dataset)
    X_anon, y_anon = preprocess_dataset(anonymized_dataset)

    # Train-test split
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(X_anon, y_anon, test_size=0.2, random_state=42)

    # Train models
    model_new = RandomForestRegressor(random_state=42)
    model_anon = LinearRegression()

    model_new.fit(X_train_new, y_train_new)
    model_anon.fit(X_train_anon, y_train_anon)

    # Evaluate models
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2

    metrics_new = evaluate_model(model_new, X_test_new, y_test_new)
    metrics_anon = evaluate_model(model_anon, X_test_anon, y_test_anon)

    st.write("### Performance Metrics")
    st.write(f"#### New Dataset:")
    st.write(f"- Mean Squared Error: {metrics_new[0]:.2f}")
    st.write(f"- Mean Absolute Error: {metrics_new[1]:.2f}")
    st.write(f"- R² Score: {metrics_new[2]:.2f}")

    st.write(f"#### Anonymized Dataset:")
    st.write(f"- Mean Squared Error: {metrics_anon[0]:.2f}")
    st.write(f"- Mean Absolute Error: {metrics_anon[1]:.2f}")
    st.write(f"- R² Score: {metrics_anon[2]:.2f}")

    st.write("### Comparison")
    st.write(f"Improvement in R²: {metrics_anon[2] - metrics_new[2]:.2f}")
    st.write(f"Change in MSE: {metrics_anon[0] - metrics_new[0]:.2f}")

    st.write("### Predict Salary")
    exp = st.slider("Experience (Years)", min_value=0, max_value=50, value=5)
    edu = st.slider("Education Level (0-10)", min_value=0, max_value=10, value=5)
    age = st.slider("Age", min_value=18, max_value=70, value=30)

    user_input = np.array([[exp, edu, age]])
    pred_salary_new = model_new.predict(user_input)[0]
    pred_salary_anon = model_anon.predict(user_input)[0]

    st.write(f"Predicted Salary (New Dataset): ${pred_salary_new:.2f}")
    st.write(f"Predicted Salary (Anonymized Dataset): ${pred_salary_anon:.2f}")
