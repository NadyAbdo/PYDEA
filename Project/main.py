import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")

import files_handling

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = files_handling.main(developer_mode=False)
df = pd.read_csv(file_path)

# Create a directory to store the PDF files if it doesn't exist
pdf_directory = os.path.join(current_directory, "Plots")
os.makedirs(pdf_directory, exist_ok=True)

def save_and_show_plot(fig, filename):
    plt.savefig(os.path.join(pdf_directory, filename + ".pdf"), bbox_inches='tight')
    plt.show()

# Data Exploration
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# Distribution Plots

#plt.subplot(1 , 3 , 1)
mu, sigma = norm.fit(df['Age'])
x = np.linspace(df['Age'].min(), df['Age'].max(), 100)
y = norm.pdf(x, mu, sigma)
plt.figure(figsize=(6, 4))
sns.distplot(df['Age'], bins=20, kde=False, fit=norm, fit_kws={"color": "C0"})
plt.plot(x, y, label=f'Normal Distribution\n$\mu={mu:.2f}$, $\sigma={sigma:.2f}$', color='C0')
#plt.xlim(df['Age'].min(), df['Age'].max())
plt.ylabel('Density')
plt.legend(loc='upper right', fontsize=9)
#plt.title('Distplot of Age')
save_and_show_plot(plt, 'Distplot of Age')



#plt.subplot(1 , 3 , 2)
mu, sigma = norm.fit(df['Annual Income (k$)'])
x = np.linspace(df['Annual Income (k$)'].min(), df['Annual Income (k$)'].max(), 100)
y = norm.pdf(x, mu, sigma)
plt.figure(figsize=(6, 4))
sns.distplot(df['Annual Income (k$)'], bins=20, kde=False, fit=norm, fit_kws={"color": "C0"})
plt.plot(x, y, label=f'Normal Distribution\n$\mu={mu:.2f}$, $\sigma={sigma:.2f}$', color='C0')
#plt.xlim(df['Annual Income (k$)'].min(), df['Annual Income (k$)'].max())
plt.ylabel('Density')
plt.legend(loc='upper right', fontsize=9)
#plt.title('Distplot of Annual Income')
save_and_show_plot(plt, 'Distplot of Annual Income')

#plt.subplot(1 , 3 , 3)
mu, sigma = norm.fit(df['Spending Score (1-100)'])
x = np.linspace(df['Spending Score (1-100)'].min(), df['Spending Score (1-100)'].max(), 100)
y = norm.pdf(x, mu, sigma)
plt.figure(figsize=(6, 4))
sns.distplot(df['Spending Score (1-100)'], bins=20, kde=False, fit=norm, fit_kws={"color": "C0"})
plt.plot(x, y, label=f'Normal Distribution\n$\mu={mu:.2f}$, $\sigma={sigma:.2f}$', color='C0')
#plt.xlim(df['Spending Score (1-100)'].min(), df['Spending Score (1-100)'].max())
plt.ylabel('Density')
plt.legend(loc='upper right', fontsize=9)
#plt.title('Distplot of Spending Score')
save_and_show_plot(plt, 'Distplot of Spending Score')

# Count Plot

plt.figure(figsize=(6, 1))
sns.countplot(y='Gender', data=df, palette=['blue', 'pink'])
save_and_show_plot(plt, 'countplot_gender')


#########################################################

# Define colors for Male and Female points
colors = {'Male': 'blue', 'Female': 'pink'}
combined_color = 'purple'

sns.regplot(x='Age', y='Annual Income (k$)', data=df[df['Gender'] == 'Male'],
            scatter_kws={'s': 50, 'color': colors['Male']},
            label='Male', fit_reg=False)
sns.regplot(x='Age', y='Annual Income (k$)', data=df[df['Gender'] == 'Female'],
            scatter_kws={'s': 50, 'color': colors['Female']},
            label='Female', fit_reg=False)
sns.regplot(x='Age', y='Annual Income (k$)', data=df,
            scatter=False, line_kws={'color': combined_color}, label='Combined')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
#plt.title('Age vs Annual Income with Combined Regression Line')
plt.legend()
save_and_show_plot(plt, 'Age vs Annual Income w.r.t Gender')

sns.regplot(x='Age', y='Spending Score (1-100)', data=df[df['Gender'] == 'Male'],
            scatter_kws={'s': 50, 'color': colors['Male']},
            label='Male', fit_reg=False)
sns.regplot(x='Age', y='Spending Score (1-100)', data=df[df['Gender'] == 'Female'],
            scatter_kws={'s': 50, 'color': colors['Female']},
            label='Female', fit_reg=False)
sns.regplot(x='Age', y='Spending Score (1-100)', data=df,
            scatter=False, line_kws={'color': combined_color}, label='Combined')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
#plt.title('Age vs Spending Score with Combined Regression Line')
plt.legend()
save_and_show_plot(plt, 'Age vs Spending Score w.r.t Gender')

sns.regplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df[df['Gender'] == 'Male'],
            scatter_kws={'s': 50, 'color': colors['Male']},
            label='Male', fit_reg=False)
sns.regplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df[df['Gender'] == 'Female'],
            scatter_kws={'s': 50, 'color': colors['Female']},
            label='Female', fit_reg=False)
sns.regplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df,
            scatter=False, line_kws={'color': combined_color}, label='Combined')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
#plt.title('Annual Income vs Spending Score with Combined Regression Line')
plt.legend()
save_and_show_plot(plt, 'Annual Income vs Spending Score w.r.t Gender')

#########################################################

# K-Means Clustering Plots

################################################

def Age_SpendingScore_Classification(X1 = df[['Age', 'Spending Score (1-100)']].iloc[:, :].values):


    #X1 = df[['Age', 'Spending Score (1-100)']].iloc[:, :].values
    inertia = []
    for n in range(1, 11):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(X1)
        inertia.append(algorithm.inertia_)

    #plt.figure(1, figsize=(15, 6))
    #plt.subplot(1 , 2 , 1)
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, 11), inertia, 'o', color = 'C0')
    plt.plot(np.arange(1, 11), inertia, '-' , alpha=0.5, color = 'C0')
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    save_and_show_plot(plt, 'kmeans_inertia_age_spending')
    # Get the default color

    default_color = plt.rcParams['lines.color']

    print("Default color:", default_color)


    algorithm = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    algorithm.fit(X1)
    labels1 = algorithm.labels_
    centroids1 = algorithm.cluster_centers_

    km = KMeans(n_clusters=4,init = 'k-means++',max_iter=300,n_init=10,random_state=0) # setting default values for max_iter and n_init
    y_means = km.fit_predict(X1)

    h = 0.02
    x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
    y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

    #plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z = Z.reshape(xx.shape)

    colors = ['magenta', 'green', 'blue', 'red']
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=cmap, aspect='auto', origin='lower', alpha=0.1)

    plt.scatter(X1[y_means==0,0],X1[y_means==0,1],s=50,c='blue',label='Standard', edgecolors='k')
    plt.scatter(X1[y_means==1,0],X1[y_means==1,1],s=50,c='green',label='Target', edgecolors='k')
    plt.scatter(X1[y_means==2,0],X1[y_means==2,1],s=50,c='red',label='Careful', edgecolors='k')
    plt.scatter(X1[y_means==3,0],X1[y_means==3,1],s=50,c='magenta',label='Sensible', edgecolors='k')
    #plt.scatter(x='Age', y='Spending Score (1-100)', data=df, c=labels1, s=200)
    plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=10, c='black', alpha=0.5,label='Centroid')
    plt.ylabel('Spending Score (1-100)')
    plt.xlabel('Age')
    plt.legend()
    save_and_show_plot(plt, 'kmeans_clusters_age_spending')

Age_SpendingScore_Classification()

################################################

def Age_AnnualIncome_Classification(X = df[['Age', 'Annual Income (k$)']].iloc[:, :].values):

    inertia = []
    for n in range(1, 11):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)

    #plt.figure(1, figsize=(15, 6))
    #plt.subplot(1 , 2 , 1)
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, 11), inertia, 'o', color = 'C0')
    plt.plot(np.arange(1, 11), inertia, '-' , alpha=0.5, color = 'C0')
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    save_and_show_plot(plt, 'kmeans_inertia_age_annual income')



    algorithm = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    algorithm.fit(X)
    labels1 = algorithm.labels_
    centroids1 = algorithm.cluster_centers_

    #X = df.iloc[:,[3,4]].values 
    km = KMeans(n_clusters=4,init = 'k-means++',max_iter=300,n_init=10,random_state=0) # setting default values for max_iter and n_init
    y_means = km.fit_predict(X)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z2 = km.predict(np.c_[xx.ravel(), yy.ravel()]) 

    #plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z2 = Z2.reshape(xx.shape)

    colors = ['blue', 'green', 'red', 'magenta']
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(Z2, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=cmap, aspect='auto', origin='lower', alpha=0.1)

    plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50,c='blue',label='Periority 1', edgecolors='k')
    plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50,c='green',label='Periority 2', edgecolors='k')
    plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50,c='red',label='Periority 3', edgecolors='k')
    plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50,c='magenta',label='Periority 4', edgecolors='k')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=10,c='black',label='Centroid')

    #plt.title('Clusters of Clients')
    plt.xlabel('Age')
    plt.ylabel('Annual Income (k$)')
    plt.legend()
    save_and_show_plot(plt, 'kmeans_clusters_age_annual income')

Age_AnnualIncome_Classification()

###########################################################################

def AnnualIncome_SpendingScore_Classification(X = df[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values):

    inertia = []
    for n in range(1, 11):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)

    #plt.figure(1, figsize=(15, 6))
    #plt.subplot(1 , 2 , 1)
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, 11), inertia, 'o', color = 'C0')
    plt.plot(np.arange(1, 11), inertia, '-' , alpha=0.5, color = 'C0')
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    save_and_show_plot(plt, 'kmeans_inertia_income_spending')



    algorithm = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    algorithm.fit(X)
    labels1 = algorithm.labels_
    centroids1 = algorithm.cluster_centers_

    #X = df.iloc[:,[3,4]].values 
    km = KMeans(n_clusters=5,init = 'k-means++',max_iter=300,n_init=10,random_state=0) # setting default values for max_iter and n_init
    y_means = km.fit_predict(X)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z2 = km.predict(np.c_[xx.ravel(), yy.ravel()]) 

    #plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z2 = Z2.reshape(xx.shape)

    colors = ['blue', 'green', 'red', 'magenta', 'brown']
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.imshow(Z2, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=cmap, aspect='auto', origin='lower', alpha=0.1)

    plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50,c='blue',label='Standard', edgecolors='k')
    plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50,c='green',label='Target', edgecolors='k')
    plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50,c='red',label='Careful', edgecolors='k')
    plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50,c='magenta',label='Sensible', edgecolors='k')
    plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50,c='brown',label='Careless', edgecolors='k')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=10,c='black',label='Centroid')

    #plt.title('Clusters of Clients')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    save_and_show_plot(plt, 'kmeans_clusters_income_spending')

AnnualIncome_SpendingScore_Classification()

