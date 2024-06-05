from statistics import mode
from typing import List, NoReturn
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import resample


class BaseRandomForest(BaseEstimator):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int or None = None,
                 min_samples_split: int = 2,
                 random_state: int or None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: List[DecisionTreeRegressor or DecisionTreeClassifier] = []

    def _fit(self, X, y, tree_type) -> NoReturn:
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y, random_state=self.random_state)
            tree = tree_type(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.swapaxes(tree_predictions, 0, 1)


class RandomForestClassifier(BaseRandomForest, ClassifierMixin):
    def fit(self, X, y):
        self._fit(X, y, DecisionTreeClassifier)

    def predict(self, X) -> List[int]:
        y_pred = self._predict(X)
        return [mode(y) for y in y_pred]


class RandomForestRegressor(BaseRandomForest, RegressorMixin):
    def fit(self, X, y):
        self._fit(X, y, DecisionTreeRegressor)

    def predict(self, X):
        y_pred = self._predict(X)
        return np.mean(y_pred, axis=1)
