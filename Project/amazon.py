import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "Files/Europe Sales Records.csv")
df = pd.read_csv(file_path)

# Drop rows with missing values
df = df.dropna()

# Data Exploration
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# Create a directory to store the PDF files if it doesn't exist
pdf_directory = os.path.join(current_directory, "Plots")
os.makedirs(pdf_directory, exist_ok=True)

def save_and_show_plot(fig, filename):
    plt.savefig(os.path.join(pdf_directory, filename + ".pdf"), bbox_inches='tight')
    plt.show()

# Example: Bar chart for Total Profit by Country
plt.figure(figsize=(15, 10))
sns.barplot(x='Country', y='Total Profit', data=df, errorbar=None)
plt.title('Total Profit by Country')
plt.xticks(rotation=90)
save_and_show_plot(plt, 'Total Profit by Country')

plt.figure(figsize=(15, 5))
sns.countplot(y='Sales Channel', data=df)
save_and_show_plot(plt, 'Sales Channel')

plt.figure(figsize=(15, 5))
sns.countplot(y='Item Type', data=df)
save_and_show_plot(plt, 'Item Type')

# Example: Time series line plot for Total Revenue over time
df['Order Date'] = pd.to_datetime(df['Order Date'])
df_time_series = df.groupby('Order Date')['Total Revenue'].sum().reset_index()
sns.lineplot(x='Order Date', y='Total Revenue', data=df_time_series)
#plt.ticklabel_format(style='plain', axis='y')
plt.title('Total Revenue Over Time')
save_and_show_plot(plt, 'Total Revenue over time')

##############################################################################


# Select relevant features for clustering (e.g., Total Revenue and Total Profit)
features_for_clustering = ['Units Sold', 'Total Profit']
data_for_clustering = df[features_for_clustering]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
save_and_show_plot(plt, 'Elbow Method for Optimal k1')

# Choose the optimal number of clusters (e.g., from the elbow method)
optimal_k = 3

# Perform k-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
sns.scatterplot(x='Units Sold', y='Total Profit', hue='Cluster', data=df, palette='viridis', s=50)
plt.title('Customer Segmentation using K-Means Clustering')
save_and_show_plot(plt, 'Customer Segmentation using K-Means Clustering 1')

################################################

# Select relevant features for clustering (e.g., Total Revenue and Total Profit)
features_for_clustering = ['Units Sold', 'Unit Cost']
data_for_clustering = df[features_for_clustering]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
save_and_show_plot(plt, 'Elbow Method for Optimal k2')

# Choose the optimal number of clusters (e.g., from the elbow method)
optimal_k = 3

# Perform k-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
sns.scatterplot(x='Units Sold', y='Unit Cost', hue='Cluster', data=df, palette='viridis', s=50)
plt.title('Customer Segmentation using K-Means Clustering')
save_and_show_plot(plt, 'Customer Segmentation using K-Means Clustering 2')

################################################

X1 = df[['Units Sold', 'Total Profit']].iloc[:, :].values
inertia = []
for n in range(1, 11):
    algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001, random_state=111, algorithm='elkan')
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

#plt.figure(1, figsize=(15, 6))
#plt.subplot(1 , 2 , 1)

plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-' , alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
save_and_show_plot(plt, 'kmeans_inertia_age_spending')



algorithm = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                   tol=0.0001, random_state=111, algorithm='elkan')
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 100
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

#plt.figure(1, figsize=(15, 7))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

plt.scatter(x='Units Sold', y='Total Profit', data=df, c=labels1, s=2)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Age')
save_and_show_plot(plt, 'kmeans_clusters_age_spending')

################################################

X1 = df[['Units Sold', 'Unit Cost']].iloc[:, :].values
inertia = []
for n in range(1, 11):
    algorithm = KMeans(n_clusters=n, random_state=24)
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

#plt.figure(1, figsize=(15, 6))
#plt.subplot(1 , 2 , 1)

plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-' , alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
save_and_show_plot(plt, 'kmeans_inertia_age_spending')



algorithm = KMeans(n_clusters=3, random_state=24)
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

h = 100
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

#plt.figure(1, figsize=(15, 7))
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

plt.scatter(x='Units Sold', y='Unit Cost', data=df, c=labels1)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Age')
save_and_show_plot(plt, 'kmeans_clusters_age_spending')

