from flask import Flask, request, jsonify, send_from_directory
import pandas as pd

app = Flask(__name__)

# Load CSV
df = pd.read_csv('food_data.csv')

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# API route for multiple foods with warnings
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    foods = data.get('foods', [])

    if not foods:
        return jsonify({"error": "No foods provided"}), 400

    results = []
    total_calories = total_protein = total_carbs = total_fat = 0
    warnings = []

    for item in foods:
        food_name = item.get('food', '').lower()
        grams = float(item.get('grams', 100))

        if grams <= 0:
            warnings.append(f"Invalid grams for '{food_name}', skipped.")
            continue

        row = df[df['food_name'].str.lower() == food_name]
        if row.empty:
            warnings.append(f"'{food_name}' not found, skipped.")
            continue

        # Calculate nutrition
        calories = row['calories_100g'].values[0] * grams / 100
        protein  = row['protein_100g'].values[0] * grams / 100
        carbs    = row['carbs_100g'].values[0] * grams / 100
        fat      = row['fat_100g'].values[0] * grams / 100

        total_calories += calories
        total_protein += protein
        total_carbs += carbs
        total_fat += fat

        results.append({
            'food': food_name,
            'grams': grams,
            'calories': round(calories, 2),
            'protein': round(protein, 2),
            'carbs': round(carbs, 2),
            'fat': round(fat, 2)
        })

    totals = {
        'total_calories': round(total_calories, 2),
        'total_protein': round(total_protein, 2),
        'total_carbs': round(total_carbs, 2),
        'total_fat': round(total_fat, 2)
    }

    return jsonify({"items": results, "totals": totals, "warnings": warnings})

if __name__ == '__main__':
    app.run(debug=True)
