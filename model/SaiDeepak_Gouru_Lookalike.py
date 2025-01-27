import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

customers = pd.read_csv('datasets/Customers.csv')
products = pd.read_csv('datasets/Products.csv')
transactions = pd.read_csv('datasets/Transactions.csv')

def create_customer_features(customers, transactions, products):
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    current_date = datetime.now()
    customers['account_age'] = (current_date - customers['SignupDate']).dt.days
    customer_features = pd.get_dummies(customers[['Region']], prefix=['region'])
    customer_features['CustomerID'] = customers['CustomerID']
    customer_features['account_age'] = customers['account_age']
    trans_prod = pd.merge(transactions, products, on='ProductID')
    category_spending = trans_prod.pivot_table(
        index='CustomerID',
        columns='Category',
        values='TotalValue',
        aggfunc='sum',
        fill_value=0
    )
    transaction_features = transactions.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'Quantity': ['sum', 'mean']
    }).fillna(0)
    transaction_features.columns = [
        'total_spend', 'avg_transaction_value', 
        'transaction_count', 'total_quantity', 'avg_quantity'
    ]
    customer_age_df = customers[['CustomerID', 'account_age']]
    transaction_features = pd.merge(
        transaction_features, 
        customer_age_df, 
        left_index=True, 
        right_on='CustomerID'
    )
    transaction_features['purchase_frequency'] = (
        transaction_features['transaction_count'] / 
        transaction_features['account_age'].clip(lower=1)
    )
    price_sensitivity = trans_prod.groupby('CustomerID').agg({
        'Price_x': 'mean',
        'Price_y': 'mean'
    }).fillna(0)
    price_sensitivity['price_sensitivity'] = (
        price_sensitivity['Price_x'] / 
        price_sensitivity['Price_y'].clip(lower=1)
    )
    feature_matrix = pd.merge(
        customer_features,
        category_spending,
        on='CustomerID',
        how='left'
    )
    feature_matrix = pd.merge(
        feature_matrix,
        transaction_features[['CustomerID', 'total_spend', 'avg_transaction_value', 
                            'transaction_count', 'purchase_frequency']],
        on='CustomerID',
        how='left'
    )
    feature_matrix = pd.merge(
        feature_matrix,
        price_sensitivity[['price_sensitivity']],
        left_on='CustomerID',
        right_index=True,
        how='left'
    )
    return feature_matrix.fillna(0)

def find_lookalikes(customer_id, feature_matrix, n_recommendations=3):
    target_features = feature_matrix[feature_matrix['CustomerID'] == customer_id]
    other_features = feature_matrix[feature_matrix['CustomerID'] != customer_id]
    target_features = target_features.drop('CustomerID', axis=1)
    other_customers = other_features['CustomerID']
    other_features = other_features.drop('CustomerID', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(other_features)
    scaled_target = scaler.transform(target_features)
    similarity_scores = cosine_similarity(scaled_target, scaled_features)[0]
    top_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
    recommendations = pd.DataFrame({
        'similar_customer_id': other_customers.iloc[top_indices].values,
        'similarity_score': similarity_scores[top_indices]
    })
    return recommendations

print("Creating feature matrix...")
feature_matrix = create_customer_features(customers, transactions, products)

print("Generating recommendations...")
results = {}
for customer_id in customers['CustomerID'][:20]:
    recommendations = find_lookalikes(customer_id, feature_matrix)
    results[customer_id] = recommendations.to_dict('records')

output_rows = []
for cust_id, recs in results.items():
    similar_customers = [f"{rec['similar_customer_id']}:{rec['similarity_score']:.3f}" for rec in recs]
    output_rows.append({
        'customer_id': cust_id,
        'lookalikes': ' | '.join(similar_customers)
    })

output_df = pd.DataFrame(output_rows)
output_df.to_csv('results/SaiDeepak_Gouru_Lookalike.csv', index=False)

print("\nSample lookalike recommendations:")
for cust_id in list(results.keys())[:5]:
    print(f"\nCustomer {cust_id}:")
    for rec in results[cust_id]:
        print(f"Similar customer: {rec['similar_customer_id']}, Score: {rec['similarity_score']:.3f}")
