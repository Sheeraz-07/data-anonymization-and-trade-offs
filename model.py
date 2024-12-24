import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def model_comparison(new_dataset, cleaned_dataset):
    st.title("Model Performance Comparison")

    # # Display first few rows of datasets
    # st.write("### New Dataset (First 10 Rows)")
    # st.dataframe(new_dataset.head(10))

    # st.write("### Cleaned Dataset (First 10 Rows)")
    # st.dataframe(cleaned_dataset.head(10))

    # Check if required columns exist
    required_columns = ['Experience', 'Salary']
    if any(col not in new_dataset.columns for col in required_columns):
        st.error("New dataset must contain 'Experience' and 'Salary' columns for modeling.")
        return
    if any(col not in cleaned_dataset.columns for col in required_columns):
        st.error("Cleaned dataset must contain 'Experience' and 'Salary' columns for modeling.")
        return

    # Prepare new_X_features and cleaned_X_features
    new_X_features = ['Names', 'Account Number', 'Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender', 'Contact', 'Email']
    cleaned_X_features = ['Experience']
    target_column = 'Salary'

    # Label encoding for categorical features in new_dataset
    le = LabelEncoder()
    for col in ['Names', 'Location', 'Job_Title', 'Gender', 'Education']:
        if col in new_dataset.columns:
            new_dataset[col] = le.fit_transform(new_dataset[col].astype(str))

    # Handle missing or malformed data by converting non-numeric columns to numeric and dropping rows with NaNs
    new_dataset = new_dataset.apply(pd.to_numeric, errors='coerce')
    cleaned_dataset = cleaned_dataset.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values from only the necessary columns ('Experience' and 'Salary')
    cleaned_dataset = cleaned_dataset.dropna(subset=['Experience', 'Salary'])

    # Ensure 'Experience' is numeric
    if cleaned_dataset['Experience'].isnull().any():
        st.error("'Experience' column contains invalid or missing values after cleaning.")
        return

    # Check if datasets have any rows left after cleaning
    if cleaned_dataset.empty:
        st.error("Cleaned dataset is empty after removing rows with missing values.")
        return

    # Scale the features for consistency
    scaler = StandardScaler()

    try:
        X_new = scaler.fit_transform(new_dataset[new_X_features])
        X_cleaned = cleaned_dataset[cleaned_X_features].values
    except ValueError as e:
        st.error(f"Error scaling data: {e}")
        return

    y_new = new_dataset[target_column].values
    y_cleaned = cleaned_dataset[target_column].values

    # Train-test split
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

    # Train models
    model_new = RandomForestRegressor(random_state=42)
    model_cleaned = LinearRegression()

    model_new.fit(X_train_new, y_train_new)
    model_cleaned.fit(X_train_cleaned, y_train_cleaned)

    # Predictions
    y_pred_new = model_new.predict(X_test_new)
    y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

    # Calculate regression metrics
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, mae, r2

    metrics_new = calculate_metrics(y_test_new, y_pred_new)
    metrics_cleaned = calculate_metrics(y_test_cleaned, y_pred_cleaned)

    # Display metrics
    def display_metrics(metrics, dataset_name):
        mse, mae, r2 = metrics
        st.write(f"### {dataset_name} Metrics")
        st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"- R² Score: {r2:.2f}")

    display_metrics(metrics_new, "Original Dataset")
    display_metrics(metrics_cleaned, "Cleaned Dataset")

    # Compare results
    st.write("### Comparison")
    st.write(f"Improvement in R² (Cleaned vs Original): {metrics_cleaned[2] - metrics_new[2]:.2f}")
    st.write(f"Change in MSE (Cleaned vs Original): {metrics_cleaned[0] - metrics_new[0]:.2f}")
