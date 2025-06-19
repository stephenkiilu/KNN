import math
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("Features and labels must have the same length")
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)

    def _euclidean_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def _predict_point(self, x):
        distances = [self._euclidean_distance(x, train_x) for train_x in self.X_train]
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return [self._predict_point(x) for x in X]

if __name__ == "__main__":
    # Simple demonstration
    X_train = [
        [1.0, 2.1], [1.5, 1.8], [5.0, 8.0],
        [6.0, 8.0], [1.0, 0.6], [9.0, 11.0]
    ]
    y_train = ['A', 'A', 'B', 'B', 'A', 'B']
    X_test = [[1.0, 1.0], [8.0, 9.0]]

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
