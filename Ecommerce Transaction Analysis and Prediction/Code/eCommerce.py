import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

# Load datasets
customers = pd.read_csv("C:/Users/vicky/Desktop/Ecommerce/Data/Customers.csv")
transactions = pd.read_csv("C:/Users/vicky/Desktop/Ecommerce/Data/Transactions.csv")
products = pd.read_csv("C:/Users/vicky/Desktop/Ecommerce/Data/Products.csv")

# Merge datasets
data = transactions.merge(customers, on='CustomerID', how='left')
data = data.merge(products, on='ProductID', how='left')

# EDA: Generate basic statistics
print(data.describe())
print(data.info())

# EDA: Plot 1 - Distribution of TotalValue
plt.figure(figsize=(8, 5))
sns.histplot(data['TotalValue'], bins=50, kde=True)
plt.title("Distribution of Transaction Values")
plt.show()

# EDA: Plot 2 - Number of Transactions per Region
plt.figure(figsize=(8, 5))
sns.countplot(x='Region', data=customers, order=customers['Region'].value_counts().index)
plt.title("Number of Customers per Region")
plt.xticks(rotation=45)
plt.show()

# EDA: Plot 3 - Top Selling Product Categories
plt.figure(figsize=(8, 5))
category_sales = data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar')
plt.title("Top Selling Product Categories")
plt.ylabel("Total Sales Value")
plt.xticks(rotation=45)
plt.show()

# Feature Engineering for Lookalike Model
customer_features = data.groupby('CustomerID').agg({
    'TotalValue': ['sum', 'mean'],
    'TransactionID': 'count'
}).reset_index()
customer_features.columns = ['CustomerID', 'Total_Spending', 'Avg_Spending', 'Transaction_Count']

# Encode categorical features
if 'Region' in customers.columns:
    le = LabelEncoder()
    customers['Region'] = le.fit_transform(customers['Region'])
    customer_features = customer_features.merge(customers[['CustomerID', 'Region']], on='CustomerID', how='left')

# Normalize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.drop(columns=['CustomerID']))

# Lookalike Model - Using Nearest Neighbors
nn = NearestNeighbors(n_neighbors=4, metric='euclidean')
nn.fit(scaled_features)
distances, indices = nn.kneighbors(scaled_features)

# Save lookalike results
lookalike_dict = {}
customer_ids = customer_features['CustomerID'].values
for i in range(20):  # First 20 customers (C0001 - C0020)
    cust_id = customer_ids[i]
    lookalikes = [(customer_ids[indices[i][j]], distances[i][j]) for j in range(1, 4)]  # Top 3 similar customers
    lookalike_dict[cust_id] = lookalikes

lookalike_df = pd.DataFrame([(k, v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1]) for k, v in lookalike_dict.items()],
                            columns=['CustomerID', 'Lookalike1', 'Score1', 'Lookalike2', 'Score2', 'Lookalike3', 'Score3'])
lookalike_df.to_csv("Lookalike.csv", index=False)

# Clustering - Apply K-Means
optimal_k = 4  # Based on Elbow Method Analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Evaluate clustering
db_index = davies_bouldin_score(scaled_features, kmeans_labels)
silhouette_avg = silhouette_score(scaled_features, kmeans_labels)

print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Add cluster labels to dataset
customer_features['Cluster'] = kmeans_labels

# Visualize clusters
sns.pairplot(customer_features, hue='Cluster', diag_kind='kde', palette='viridis')
plt.show()

# Save clustering results
customer_features.to_csv("Clustered_Customers.csv", index=False)
