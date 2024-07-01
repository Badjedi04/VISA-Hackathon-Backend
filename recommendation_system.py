import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

data = pd.read_pickle('cleaned_data.pkl')

numerical_cols = [
    'energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 
    'proteins_100g', 'salt_100g', 'sodium_100g', 'energy-kj_100g', 'energy-kcal_100g', 
    'saturated-fat_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 
    'carbon-footprint-from-meat-or-fish_100g'
]
categorical_cols = [
    'quantity', 'brands', 'categories_en', 'allergens', 'stores', 'countries_en', 
    'ingredients_analysis_tags', 'traces_en', 'serving_size', 'nutrient_levels_tags', 
    'ecoscore_grade', 'nutriscore_grade'
]


embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def find_closest_product(user_input):
    user_embedding = embedding_model.encode(user_input)
    similarities = cosine_similarity([user_embedding], np.vstack(data['product_name_embedding'].values))
    closest_idx = similarities.argmax()
    return data.iloc[closest_idx]

def recommend_products(product_features):
    product_features = np.nan_to_num(product_features) 
    
    similarities = cosine_similarity([product_features], data[numerical_cols].values)
    data['similarity'] = similarities[0]
    
    filtered_data = data[(data['ecoscore_grade'] <= 3) & (data['nutriscore_grade'] <= 3.0)]  
    filtered_data = filtered_data.sort_values(by=['ecoscore_grade', 'nutriscore_grade', 'similarity'], ascending=[True, True, False])
    
    top_products = filtered_data.head(5)
    return top_products

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')  
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_description(product_name):
    prompt = f"Explain the environmental benefits and sustainability features of {product_name} to help consumers make eco-friendly choices and reduce their environmental footprint:"
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#     inputs = tokenizer(prompt, return_tensors='pt', padding=True)
#     outputs = model.generate(
#       inputs['input_ids'], 
#       max_length=100, 
#        num_return_sequences=1, 
#        pad_token_id=tokenizer.eos_token_id, 
#        attention_mask=inputs['attention_mask']
#    )
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    response_part = description.split("footprint:")[1].strip().replace("\n", "")

  
    return response_part