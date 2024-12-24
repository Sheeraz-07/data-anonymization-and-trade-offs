
import pandas as pd
import re

def identify_sensitive_columns(data):
    sensitive_keywords = [
        "name", "full_name", "first_name", "last_name", "account", "contact", "email", 
        "location", "age", "gender", "job", "title", "phone", "address", "dob", "date_of_birth", 
        "ssn", "credit_card", "pin", "birthdate", "email_id", "passport", "id", "user_id", 
        "username", "employee", "salesman", "representative", "manager", "staff", "person", 
        "national_id", "social_security", "zipcode", "bank_account", "mobile", "worker", 
        "individual", "profile", "personal_id", "driver_license", "tax_id"
    ]

    def calculate_probability(column_name, column_data):
        score = 0

        for keyword in sensitive_keywords:
            if re.search(keyword, column_name.lower()):
                score += 0.5

        sample_data = column_data.sample(min(len(column_data), 50), random_state=42).astype(str)
        for value in sample_data:
            if re.match(r"^[\w\.\-]+@[\w\.\-]+\.\w{2,4}$", value):  # Email pattern
                score += 0.3
            elif re.match(r"^\+?\d{10,15}$", value):  # Phone number pattern
                score += 0.3
            elif re.match(r"^[A-Za-z]+(\s[A-Za-z]+)*$", value):  # Name pattern (single or full)
                score += 0.2
            elif re.match(r"^[0-9]{13,16}$", value):  # Account number pattern
                score += 0.2
            elif re.match(r"^\d{4}-\d{2}-\d{2}$", value):  # Date pattern (YYYY-MM-DD)
                score += 0.1
            elif value.isdigit() and 0 <= int(value) <= 100:  # Age range check
                score += 0.1
            elif value.lower() in ["male", "female", "other", "gender"]:  # Gender check
                score += 0.2

        return score

    sensitive_columns = {}
    for column in data.columns:
        if 'experience' in column.lower() or 'salary' in column.lower():
            continue

        keyword_detected = False
        for keyword in sensitive_keywords:
            if re.search(keyword, column.lower()):
                sensitive_columns[column] = "Detected by Keyword"
                keyword_detected = True
                break

        if not keyword_detected:
            score = calculate_probability(column, data[column])
            if score >= 0.7:
                sensitive_columns[column] = score

    return sensitive_columns

def remove_sensitive_columns(data, sensitive_columns):
    non_sensitive_columns = [col for col in data.columns if col not in sensitive_columns]
    return data[non_sensitive_columns]

def clean_dataset(data):
    sensitive_columns = identify_sensitive_columns(data)
    cleaned_data = remove_sensitive_columns(data, sensitive_columns)
    return sensitive_columns, cleaned_data
