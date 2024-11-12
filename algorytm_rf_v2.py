import random

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=2):
        samples_number, features_number = X.shape
        if (samples_number >= self.min_samples_split) and (depth < self.max_depth):
            best_split = self._best_split(X, y, features_number)
            if best_split["gain"] > 0:
                left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return {"feature_index": best_split["feature_index"],
                        "threshold": best_split["threshold"],
                        "left": left_subtree,
                        "right": right_subtree}
        return self._leaf_node(y)

    def _best_split(self, X, y, features_number):
        best_split = {}
        best_gain = -float("inf")
        for feature_index in range(features_number):
            thresholds = set(X[:, feature_index])
            for threshold in thresholds:
                gain, X_left, y_left, X_right, y_right = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "X_left": X_left,
                        "y_left": y_left,
                        "X_right": X_right,
                        "y_right": y_right,
                        "gain": gain
                    }
        return best_split

    def _information_gain(self, X, y, feature_index, threshold):
        left_indexes = X[:, feature_index] <= threshold
        right_indexes = X[:, feature_index] > threshold
        if sum(left_indexes) == 0 or sum(right_indexes) == 0:
            return 0, None, None, None, None

        y_left, y_right = y[left_indexes], y[right_indexes]
        p_left, p_right = len(y_left) / len(y), len(y_right) / len(y)
        gain = self._gini_impurity(y) - (p_left * self._gini_impurity(y_left) + p_right * self._gini_impurity(y_right))
        return gain, X[left_indexes], y[left_indexes], X[right_indexes], y[right_indexes]

    def _gini_impurity(self, y):
        proportions = [sum(y == c) / len(y) for c in set(y)]
        return 1 - sum([p ** 2 for p in proportions])

    def _leaf_node(self, y):
        return {"value": max(set(y), key=list(y).count)}

    def _traverse_tree(self, x, node):
        if "value" in node:
            return node["value"]
        feature_value = x[node["feature_index"]]
        if feature_value <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]


class RandomForest:
    def __init__(self, n_trees=20, max_depth=3, max_features=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._rf_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _rf_sample(self, X, y):
        n_samples = X.shape[0]
        indexes = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return X[indexes], y[indexes]

    def predict(self, X):
        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = list(zip(*tree_preds))
        return [max(set(preds), key=preds.count) for preds in tree_preds]

#-------------------------------------------------------------------Testowanie------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score
#przykladowe dane
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
#etykiety
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

forest = RandomForest()
forest.fit(X, y)

#stare testy
#X_test = np.array([[1,7], [8, 11], [2,12], [2,5], [1,5], [6,5], [2,2], [7,4], [2,1], [8,3]]) #dane testowe
#prediction = forest.predict(X_test)
#prediction = np.array(prediction).astype(int)
#print("\nPredykcja z danych wpisanych:", prediction)
#print("Etykiety wpisane: ", y)

#---------------------------------------------------------------------------------MNIST---------------------------------
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train / 255.0
X_test = X_test / 255.0

from sklearn.decomposition import PCA

pca = PCA(n_components=60)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

forest = RandomForest(n_trees=10, max_depth=2)
forest.fit(X_train_pca, y_train)

#Predykcja na danych mnist
predictions = forest.predict(X_test_pca[:10])
acc_mnist = accuracy_score(y_test[:10], predictions)
predictions = np.array(predictions).astype(int)
print("\nPredykcje mnist:", predictions)
print("Etykiety mnist:", y_test[:10])
print("Dokładność MNIST: ",acc_mnist,'\n')

#------------------------------------------------------------------------------------IRIS-------------------------------
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

pca_iris = PCA(n_components=2)

X_train_pca_iris = pca_iris.fit_transform(X_train_iris)
X_test_pca_iris = pca_iris.transform(X_test_iris)

forest_iris = RandomForest(n_trees=20, max_depth=5)

forest_iris.fit(X_train_pca_iris, y_train_iris)

predictions_iris = forest_iris.predict(X_test_pca_iris)
acc_iris = accuracy_score(y_test_iris, predictions_iris)
predictions_iris = np.array(predictions_iris).astype(int)

print("\nPredykcje Iris:", predictions_iris)
print("Etykiety Iris:", y_test_iris)
print("Dokładność Iris:", acc_iris)
