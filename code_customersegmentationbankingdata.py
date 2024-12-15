# Below implementation of the customer segmentation analysis covers all the main aspects of the analysis, including:
# 1.Data loading and preprocessing
# 2.Exploratory Data Analysis (EDA)
# 3.Feature engineering and normalization
# 4.Principal Component Analysis (PCA)
# 5.K-means clustering
# 6.UMAP visualization
# 7.Cluster analysis
# 8.Customer persona creation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
customer_df = pd.read_csv("/Users/supriyakushwaha/Desktop/Projects/Customer_Segmentation_Banking/Marketing_data.csv")


# Clean column names
customer_df.columns = customer_df.columns.str.lower().str.replace(' ', '_')

# Handle missing values
customer_df['minimum_payments'] = customer_df['minimum_payments'].fillna(customer_df['minimum_payments'].median())
customer_df['credit_limit'] = customer_df['credit_limit'].fillna(customer_df['credit_limit'].median())

# Create credit utilization feature
customer_df['credit_utilization'] = customer_df['balance'] / customer_df['credit_limit']

# Filter for customers with tenure of 10 months
df_filtered = customer_df[customer_df['tenure'] == 10].copy()

# Drop unnecessary columns
columns_to_drop = ['cust_id', 'installments_purchases']
df_filtered = df_filtered.drop(columns=columns_to_drop)

# Exploratory Data Analysis functions
def plot_histograms(df):
    n_cols = len(df.columns)
    n_rows = (n_cols + 3) // 4  # Round up to the nearest multiple of 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes
    
    for i, col in enumerate(df.columns):
        if i < len(axes):
            sns.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(col)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_credit_utilization(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['credit_utilization'] <= 1]['credit_utilization'], kde=True)
    plt.title('Credit Utilization Distribution (<=100%)')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['credit_utilization'] > 1]['credit_utilization'], kde=True)
    plt.title('Credit Utilization Distribution (>100%)')
    plt.show()

def plot_cash_advance_cohorts(df):
    df['cash_advance_cohort'] = pd.cut(df['cash_advance'], 
                                       bins=[-np.inf, 1000, 5000, np.inf], 
                                       labels=['Low', 'Medium', 'High'])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='cash_advance_cohort', y='cash_advance', data=df[df['purchases'] == 0])
    plt.title('Average Cash Advance by Cohort (Customers with Zero Purchases)')
    plt.show()

# Call the EDA functions
plot_histograms(df_filtered)
plot_credit_utilization(df_filtered)
plot_cash_advance_cohorts(df_filtered)

# Feature Engineering and Normalization
features = df_filtered.columns.drop(['tenure', 'cash_advance_cohort'])
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_filtered[features]), columns=features)

# Apply PCA
pca = PCA(n_components=0.8, random_state=42)
df_pca = pd.DataFrame(pca.fit_transform(df_normalized))

# Determine optimal number of clusters
def plot_elbow_curve(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

plot_elbow_curve(df_pca)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(df_pca)

# Visualize clusters using UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(df_pca)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df_filtered['cluster'], cmap='viridis')
plt.colorbar(scatter)
plt.title('UMAP projection of the clusters')
plt.show()

# Analyze clusters
def plot_feature_distribution(df, features):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        sns.boxplot(x='cluster', y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by Cluster')
    
    plt.tight_layout()
    plt.show()

features_to_analyze = ['balance', 'cash_advance', 'credit_utilization', 'purchases']
plot_feature_distribution(df_filtered, features_to_analyze)

features_to_analyze = ['credit_limit', 'minimum_payments', 'payments', 'purchases_trx']
plot_feature_distribution(df_filtered, features_to_analyze)

# Create customer personas
def create_persona(cluster_data):
    persona = {
        'credit_utilization': f"{cluster_data['credit_utilization'].median():.2f} - {cluster_data['credit_utilization'].max():.2f}",
        'cash_advance': f"${cluster_data['cash_advance'].median():.0f} - ${cluster_data['cash_advance'].max():.0f}",
        'purchases': f"${cluster_data['purchases'].median():.0f} - ${cluster_data['purchases'].max():.0f}",
        'balance': f"${cluster_data['balance'].median():.0f} - ${cluster_data['balance'].max():.0f}"
    }
    return persona

personas = {}
for cluster in df_filtered['cluster'].unique():
    cluster_data = df_filtered[df_filtered['cluster'] == cluster]
    personas[f"Cluster {cluster}"] = create_persona(cluster_data)

for cluster, persona in personas.items():
    print(f"{cluster}:")
    for key, value in persona.items():
        print(f"  {key}: {value}")
    print()
