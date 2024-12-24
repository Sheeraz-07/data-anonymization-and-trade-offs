import pandas as pd
from faker import Faker
import random

# Load the dataset
df = pd.read_csv('new_dataset.csv')

# Initialize Faker
fake = Faker()
Faker.seed(42)  # Seed for reproducibility

# Generate unique contact numbers
num_rows = len(df)  # Use the number of rows in your dataset
base_number = "+92302"
contact_numbers = []
while len(contact_numbers) < num_rows:
    number = f"{base_number}{random.randint(1000000, 9999999)}"
    if number not in contact_numbers:
        contact_numbers.append(number)

# Generation of account numbers
account_numbers = [f"sn{random.randint(10**13, 10**14 - 1)}" for _ in range(num_rows)]

# Function to generate email with custom domain
def generate_email_with_gmail():
    username = fake.user_name()  # Generate a random username
    return f"{username}@gmail.com"  # Append @gmail.com to the username

# Replace sensitive columns with synthetic values
def replace_sensitive_data(row):
    index = row.name  # Get the index of the current row
    row['Names'] = fake.name()
    row['Account Number'] = account_numbers[index]
    row['Contact'] = contact_numbers[index]  # Assign a single contact number
    row['Email'] = generate_email_with_gmail()  # Use custom email function
    return row

df = df.apply(replace_sensitive_data, axis=1)

# Save the anonymized dataset
output_path = 'anonymized_dataset.csv'  # Adjusted for Colab's file system
df.to_csv(output_path, index=False)

print(f"Anonymized dataset saved to {output_path}")
print(df.tail(10))
