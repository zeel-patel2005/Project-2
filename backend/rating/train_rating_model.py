import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("C:/Users/zeelp/Desktop/Project-2/data/zomato.csv", encoding='latin1')

# Select relevant features
df = df[['location', 'approx_cost(for two people)', 'cuisines', 'rest_type', 'book_table', 'rate']]

# Clean 'rate' column (remove '/5' and convert to float)
df['rate'] = df['rate'].str.replace('/5', '', regex=False)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean 'approx_cost(for two people)' column (remove commas and convert to float)
df['approx_cost(for two people)'] = (
    df['approx_cost(for two people)']
    .astype(str)
    .str.replace(',', '', regex=False)
)
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
le_location = LabelEncoder()
le_cuisines = LabelEncoder()
le_rest_type = LabelEncoder()
le_book_table = LabelEncoder()

df['location'] = le_location.fit_transform(df['location'])
df['cuisines'] = le_cuisines.fit_transform(df['cuisines'])
df['rest_type'] = le_rest_type.fit_transform(df['rest_type'])
df['book_table'] = le_book_table.fit_transform(df['book_table'])

# Features & target
X = df[['location', 'approx_cost(for two people)', 'cuisines', 'rest_type', 'book_table']]
y = df['rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_location, open("le_location.pkl", "wb"))
pickle.dump(le_cuisines, open("le_cuisines.pkl", "wb"))
pickle.dump(le_rest_type, open("le_rest_type.pkl", "wb"))
pickle.dump(le_book_table, open("le_book_table.pkl", "wb"))

print("Model trained and saved!")
