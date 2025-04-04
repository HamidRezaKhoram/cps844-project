import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

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

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
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
def feature_selection(X_train, X_test, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    return X_train_new, X_test_new

# 1. Decision Tree
def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {score:.4f}")

# 2. K-Nearest Neighbors
def knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"K-Nearest Neighbors Accuracy: {score:.4f}")

# 3. Gaussian Naive Bayes
def gaussian_nb(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Gaussian Naive Bayes Accuracy: {score:.4f}")

def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

# 5. K-Means Clustering
def k_means_clustering(X_train, X_test, y_train, y_test):
    clf = KMeans(n_clusters=7, random_state=42)  # We use 7 clusters as an example
    clf.fit(X_train)
    print(f"K-Means Clustering Labels on X_train: {clf.labels_[:10]}")  # Display first 10 cluster labels

# Function to test with and without feature selection
def test_classifiers(X_train, X_test, y_train, y_test):
    # Without feature selection
    print("\nWithout Feature Selection:")
    decision_tree(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    gaussian_nb(X_train, X_test, y_train, y_test)
    k_means_clustering(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)

    # With feature selection
    print("\nWith Feature Selection:")
    X_train_new, X_test_new = feature_selection(X_train, X_test)
    decision_tree(X_train_new, X_test_new, y_train, y_test)
    knn(X_train_new, X_test_new, y_train, y_test)
    gaussian_nb(X_train_new, X_test_new, y_train, y_test)
    k_means_clustering(X_train_new, X_test_new, y_train, y_test)
    random_forest(X_train_new, X_test_new, y_train, y_test)
# Run the tests
test_classifiers(X_train, X_test, y_train, y_test)
