import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    feature_matrix = pd.merge(
        customer_features,
        transaction_features,
        on='CustomerID',
        how='left'
    )
    return feature_matrix.fillna(0)

feature_matrix = create_customer_features(customers, transactions, products)

features_for_clustering = feature_matrix.drop('CustomerID', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

max_clusters = 10
db_scores = []
inertias = []
silhouette_scores = []

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    db_scores.append(davies_bouldin_score(scaled_features, kmeans.labels_))
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(2, max_clusters + 1), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 3, 2)
plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index')

plt.subplot(1, 3, 3)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.savefig('results/clustering_metrics.png')
plt.close()

optimal_clusters = db_scores.index(min(db_scores)) + 2
print(f"\nOptimal number of clusters: {optimal_clusters}")
print(f"Best DB Index: {min(db_scores):.4f}")
print(f"Best Silhouette Score: {max(silhouette_scores):.4f}")

final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = final_kmeans.fit_predict(scaled_features)

feature_matrix['Cluster'] = cluster_labels

cluster_profiles = []
for cluster in range(optimal_clusters):
    cluster_data = feature_matrix[feature_matrix['Cluster'] == cluster]
    profile = {
        'Cluster_Size': len(cluster_data),
        'Avg_Total_Spend': cluster_data['total_spend'].mean(),
        'Avg_Transaction_Value': cluster_data['avg_transaction_value'].mean(),
        'Avg_Purchase_Frequency': cluster_data['purchase_frequency'].mean()
    }
    cluster_profiles.append(profile)

cluster_summary = pd.DataFrame(cluster_profiles)
print("\nCluster Profiles:")
print(cluster_summary)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments Visualization (PCA)')
plt.colorbar(scatter, label='Cluster')
plt.savefig('results/cluster_visualization.png')
plt.close()

feature_matrix.to_csv('results/customer_segments.csv', index=False)
cluster_summary.to_csv('results/cluster_profiles.csv', index=False)
