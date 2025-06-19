# KNN from Scratch

This repository contains a minimal implementation of the **K-Nearest Neighbors** (KNN) classification algorithm written entirely in Python. The goal is to provide a clear and easy to follow example of how KNN works without relying on external machine learning libraries.

## Overview

KNN is a simple yet powerful technique for classifying data points based on the classes of their nearest neighbors. When making a prediction for a new sample, the algorithm:

1. Computes the distance between the new sample and all points in the training set.
2. Selects the `k` closest samples.
3. Assigns the most common label among these neighbors to the new sample.

The implementation in this repository uses the Euclidean distance metric and supports any numeric feature set.

## File Structure

- `knn.py` – Implements the `KNNClassifier` class and provides a small demonstration when run as a script.
- `README.md` – Documentation and usage instructions.

## Running the Example

The classifier does not require external dependencies and runs with the default Python installation. To see it in action, execute:

```bash
python3 knn.py
```

The script trains a small toy dataset and prints the predicted labels for two test points.

## Using `KNNClassifier` in Your Project

Below is a minimal example of how to integrate the classifier in your own code:

```python
from knn import KNNClassifier

# Prepare your training data (list of feature vectors) and labels
X_train = [[1.0, 2.1], [1.5, 1.8], [5.0, 8.0]]
y_train = ['A', 'A', 'B']

# Create the classifier with k neighbors
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# Predict labels for new samples
X_test = [[1.0, 1.0], [8.0, 9.0]]
labels = knn.predict(X_test)
print(labels)
```

### Tuning Parameters

- **`k`** – The number of neighbors to consider. Increasing `k` tends to smooth the decision boundary but may reduce sensitivity to local patterns.
- **Distance metric** – By default, Euclidean distance is used. You can modify `_euclidean_distance` in `knn.py` if a different metric is needed.

## Algorithm Details

For each prediction, the classifier computes the Euclidean distance between the test point and every training point. The indices of the `k` smallest distances are used to retrieve the corresponding labels. The final prediction is the label that occurs most frequently among these neighbors.

Although this implementation is intentionally simple, it can serve as a starting point for experimenting with more advanced features such as weighted voting, different distance metrics, or support for regression problems.

## Contributing

Contributions that improve the clarity of the code or documentation are welcome. Feel free to open issues or pull requests if you have suggestions.

## License

This project is released under the MIT License.
