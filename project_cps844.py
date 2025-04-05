import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Load the data
pd.set_option('display.max_columns', None)
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
# print out the type of data
print(data.dtypes)

# Preprocessing: Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

# missing data
num_missing_data = data.isnull().sum()
print(num_missing_data)

# Encode categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Standardizing
scaler = StandardScaler()
data[['Age', 'Height', 'Weight']] = scaler.fit_transform(data[['Age', 'Height', 'Weight']])
print(data.head())

# Features and target
X = data.drop(columns=['NObeyesdad'])
y = data['NObeyesdad']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )

# Visualize the data and explore relationships
def perform_eda():
    # Load the dataset
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # General distribution (mean and standard deviation)
    print("Mean and Standard Deviation:\n", data[numerical_features].agg(['mean', 'std']))

    # Boxplots
    plt.figure(figsize=(15, len(numerical_features) * 3))
    for i, col in enumerate(numerical_features):
        plt.subplot(len(numerical_features), 1, i + 1)
        sns.boxplot(x=data[col], color='skyblue')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    # Histograms
    plt.figure(figsize=(15, len(numerical_features) * 3))
    for i, col in enumerate(numerical_features):
        plt.subplot(len(numerical_features), 1, i + 1)
        sns.histplot(data[col], kde=True, bins=20, color='green')
        plt.title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()

    # Correlation heatmap
    corr_matrix = data[numerical_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

    # Distribution of target variable
    plt.figure(figsize=(8, 5))
    sns.countplot(y='NObeyesdad', data=data, order=data['NObeyesdad'].value_counts().index, palette='viridis')
    plt.title('Distribution of Obesity Levels')
    plt.xlabel('Number of Individuals')
    plt.ylabel('Obesity Level')
    plt.tight_layout()
    plt.show()

# Example usage:
# perform_eda()

# Feature selection (for use with each classifier)

def feature_selection(X_train, X_test, y_train,
                      var_threshold=0.01,  # variance filter cutoff
                      n_features_to_select=3):  # how many features you want
    """
    1. Removes features with variance below 'var_threshold'
    2. Applies RFE (Recursive Feature Elimination) with a chosen estimator
    Returns:
      X_train_new, X_test_new - The transformed training and testing sets
    """
    # 1. Variance Threshold to drop near-constant features
    vt = VarianceThreshold(threshold=var_threshold)
    X_train_var = vt.fit_transform(X_train)
    X_test_var = vt.transform(X_test)

    # 2. RFE with an estimator
    #    LogisticRegression is a common choice for RFE.
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    rfe_selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    X_train_new = rfe_selector.fit_transform(X_train_var, y_train)
    X_test_new = rfe_selector.transform(X_test_var)

    return X_train_new, X_test_new

# 1. Decision Tree
def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {score:.4f}")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 2. K-Nearest Neighbors
def knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"K-Nearest Neighbors Accuracy: {score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 3. Gaussian Naive Bayes
def gaussian_nb(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Gaussian Naive Bayes Accuracy: {score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))
# Random Forest
def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. K-Means Clustering
def k_means_clustering(X_train, X_test, y_train, y_test):
    """
    Fits K-Means on the training set, then compares cluster assignments
    to actual class labels by mapping each cluster to its majority class.
    Prints confusion matrices and accuracy for both training and testing sets.
    """
    # Use the number of unique classes in y_train as the number of clusters
    n_clusters = y_train.nunique()

    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)

    # -----------------------------
    # Step 1: Map cluster -> majority class on the training data
    # -----------------------------
    train_cluster_labels = kmeans.labels_  # cluster assignments for X_train
    label_map = {}

    for cluster_id in np.unique(train_cluster_labels):
        # Indices where X_train belongs to cluster_id
        cluster_indices = np.where(train_cluster_labels == cluster_id)[0]
        # Actual classes for these indices
        cluster_classes = y_train.iloc[cluster_indices]
        # Find the majority class for this cluster
        majority_class = cluster_classes.value_counts().idxmax()
        label_map[cluster_id] = majority_class


    # Predict cluster assignments for X_test
    test_cluster_labels = kmeans.predict(X_test)
    # Map those cluster assignments to the majority class
    mapped_preds_test = [label_map[cluster_id] for cluster_id in test_cluster_labels]

    # Accuracy and Confusion Matrix on test data
    test_accuracy = accuracy_score(y_test, mapped_preds_test)
    test_cm = confusion_matrix(y_test, mapped_preds_test)

    print(f"K-Means Clustering (Test) Accuracy: {test_accuracy:.4f}")
    print("Test Confusion Matrix:\n", test_cm)

# Function to test with and without feature selection
def test_classifiers(X_train, X_test, y_train, y_test):
    # Without feature selection
    print("\nWithout Feature Selection:")
    decision_tree(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    gaussian_nb(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    k_means_clustering(X_train, X_test, y_train, y_test)


    # With feature selection
    print("\nWith Feature Selection:")
    X_train_new, X_test_new = feature_selection(X_train, X_test, y_train)
    decision_tree(X_train_new, X_test_new, y_train, y_test)
    knn(X_train_new, X_test_new, y_train, y_test)
    gaussian_nb(X_train_new, X_test_new, y_train, y_test)
    random_forest(X_train_new, X_test_new, y_train, y_test)
    k_means_clustering(X_train_new, X_test_new, y_train, y_test)

# Run the tests
test_classifiers(X_train, X_test, y_train, y_test)
