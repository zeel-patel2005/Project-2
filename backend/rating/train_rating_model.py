import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("C:/Users/zeelp/Desktop/research p 2/data/zomato.csv", encoding='latin1')

# Drop unnecessary columns
df = df.drop(['dish_liked', 'phone', 'reviews_list', 'menu_item', 'listed_in(city)'], axis=1, errors='ignore')

# Drop rows with missing values in target column
df = df[df['rate'].notnull()]

# Clean and convert 'rate' column
df['rate'] = df['rate'].astype(str).str.strip().str.split('/').str[0]
df = df[~df['rate'].isin(['NEW', '-', ''])]  # Remove invalid entries
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean 'votes'
df['votes'] = df['votes'].astype(str).str.replace(',', '', regex=False)
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Clean cost column
df['approx_cost'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '', regex=False)
df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')

# Drop rows with any NaN after cleaning
df = df.dropna()

# Select final feature columns
features = ['online_order', 'book_table', 'location', 'cuisines', 'votes', 'approx_cost']
df = df[features + ['rate']]

# Encode binary Yes/No columns
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
le_location = LabelEncoder()
le_cuisines = LabelEncoder()

df['location'] = le_location.fit_transform(df['location'].astype(str).str.strip())
df['cuisines'] = le_cuisines.fit_transform(df['cuisines'].astype(str).str.strip())

# Prepare X and y
X = df[features]
y = df['rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('location_encoder.pkl', 'wb') as f:
    pickle.dump(le_location, f)

with open('cuisines_encoder.pkl', 'wb') as f:
    pickle.dump(le_cuisines, f)

print("✅ Model training complete and saved.")
