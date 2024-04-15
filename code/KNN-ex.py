import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        # Get labels 
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Get min distance in number of K points.
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
if __name__ == "__main__":
    dataset = pd.read_csv('./Iris.csv')
    X = dataset.iloc[:, [1, 2, 3, 4]].values
    y = dataset.iloc[:, -2].values
    k_folds = 10
    fold_size = 10
    print("Group 3 -- KNN result\n")
    for fold in range(k_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        # Implement KNN 
        model = KNN(k=5)
        model.fit(X_train, y_train)
        # predict label 
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Fold {fold+1}, Accuracy: {accuracy}")
