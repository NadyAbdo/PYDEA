import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwopts) -> GridSearchCV:
    model = GridSearchCV(base_model, paramgrid, cv=cv, n_jobs=-1, **kwopts)
    model.fit(features, targets)
    return model

def plot_evaluation(ax, model, X_test, y_test, title, target_names):
    h = .02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    unique_classes = np.unique(y_test)
    for i, class_label in enumerate(unique_classes):
        class_indices = np.where(y_test == class_label)
        ax.scatter(X_test[class_indices, 0], X_test[class_indices, 1], label=f'Iris.{target_names[class_label]}', edgecolors='k', marker='o')

    ax.set_title(title)

    ax.legend(loc='upper right')

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # first two columns
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    target_names = iris.target_names

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    # KNN
    knn_param_grid = {'n_neighbors': np.arange(1, 50)}
    knn_model = find_hyperparams(KNeighborsClassifier(), knn_param_grid, X_train, y_train)
    knn_accuracy = knn_model.score(X_test, y_test)
    plot_evaluation(axs[0], knn_model, X_test, y_test, f'KNN - Accuracy: {knn_accuracy:.2f}\n{knn_model.best_params_}', target_names)

    # SVM with linear kernel
    svm_linear_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear']}
    svm_linear_model = find_hyperparams(SVC(), svm_linear_param_grid, X_train, y_train)
    svm_linear_accuracy = svm_linear_model.score(X_test, y_test)
    plot_evaluation(axs[1], svm_linear_model, X_test, y_test, f'SVM Linear - Accuracy: {svm_linear_accuracy:.2f}\n{svm_linear_model.best_params_}', target_names)

    # SVM with polynomial kernel
    svm_poly_param_grid = {'C': [0.1, 1, 10], 'coef0': np.arange(1, 5), 'degree': np.arange(2, 6), 'kernel': ['poly']}
    svm_poly_model = find_hyperparams(SVC(), svm_poly_param_grid, X_train, y_train)
    svm_poly_accuracy = svm_poly_model.score(X_test, y_test)
    plot_evaluation(axs[2], svm_poly_model, X_test, y_test, f'SVM Polynomial - Accuracy: {svm_poly_accuracy:.2f}\n{svm_poly_model.best_params_}', target_names)

    # SVM with radial basis function (RBF) kernel
    svm_rbf_param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf']}
    svm_rbf_model = find_hyperparams(SVC(), svm_rbf_param_grid, X_train, y_train)
    svm_rbf_accuracy = svm_rbf_model.score(X_test, y_test)
    plot_evaluation(axs[3], svm_rbf_model, X_test, y_test, f'SVM RBF - Accuracy: {svm_rbf_accuracy:.2f}\n{svm_rbf_model.best_params_}', target_names)

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()
