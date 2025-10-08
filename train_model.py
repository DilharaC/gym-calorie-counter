import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv('food_data.csv')
X = df['food_name'].str.lower()
y = df[['calories_100g','protein_100g','carbs_100g','fat_100g']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),
                  ('rf', RandomForestRegressor(n_estimators=200, random_state=42))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print("MAE (cal/protein/carbs/fat):", mae)
joblib.dump(model, 'calorie_model.pkl')
print("✅ Model saved as calorie_model.pkl")
