# First, install required libraries:
# pip install faker pandas

from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Set seed for reproducible data
Faker.seed(42)
random.seed(42)

# Define number of records
num_records = 1000

# Generate sample data
data = []

for i in range(num_records):
    record = {
        'customer_id': i + 1,
        'first_name': fake.first_name(),
        'last_name': fake.last_name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'age': random.randint(18, 80),
        'city': fake.city(),
        'country': fake.country(),
        'job_title': fake.job(),
        'salary': random.randint(30000, 150000),
        'registration_date': fake.date_between(start_date='-2y', end_date='today'),
        'last_login': fake.date_time_between(start_date='-30d', end_date='now'),
        'orders_count': random.randint(0, 50),
        'total_spent': round(random.uniform(0, 5000), 2),
        'customer_segment': random.choice(['Bronze', 'Silver', 'Gold', 'Platinum']),
        'is_active': random.choice([True, False]),
        'department': random.choice(['IT', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations'])
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sample_customer_data.csv', index=False)

print(f"Generated CSV file with {num_records} records!")
print(f"Columns: {list(df.columns)}")
print(f"File saved as: sample_customer_data.csv")

# Display first few rows
print("\nFirst 5 rows preview:")
print(df.head())