from flask import Flask, request, jsonify
from recommendation_system import data, numerical_cols, recommend_products, generate_description, find_closest_product
import numpy as np 

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        grocery_item = request.json['grocery_item']
        print(f"Received grocery item: {grocery_item}")

        closest_product = find_closest_product(grocery_item)
        print(f"Closest product found: {closest_product['product_name']}")

        product_features = closest_product[numerical_cols].values
        product_features = np.nan_to_num(product_features)  

        recommended_products = recommend_products(product_features)

        recommendations = []
        for _, recommended_product in recommended_products.iterrows():
            description = generate_description(recommended_product['product_name'])
            recommendations.append({
                'product_name': recommended_product['product_name'],
                'ecoscore_grade': recommended_product['ecoscore_grade'],
                'nutriscore_grade': recommended_product['nutriscore_grade'],
                'description': description
            })

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)