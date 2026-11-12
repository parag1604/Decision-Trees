"""Decision trees (CART) from scratch: a classification tree and a regression tree.

Both trees are binary: every internal node asks "is feature j < threshold?",
sends the point left or right, and every leaf stores a decision (the majority
label for classification, the mean target for regression). Trees are grown
greedily: at each node we try every feature and every threshold (midpoints
between consecutive unique values) and keep the split that reduces the
impurity the most.
"""
import numpy as np
from typing import Tuple, Optional


class DTNode:
    """A node of a classification tree (the root node is the whole tree)."""

    def __init__(
            self,
            info_criteria: str = "entropy",
            max_depth: Optional[int] = None,
    ) -> None:
        self.category_idx: int = -1     # feature used for the split
        self.threshold: float = -1      # go left if x[category_idx] < threshold
        self.max_depth = max_depth      # remaining depth budget (None = unlimited)
        self.left: Optional[DTNode] = None
        self.right: Optional[DTNode] = None
        self.decision = None            # majority label of the training points here
        self.info_criteria = info_criteria

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def _get_entropy(
            self,
            y: np.ndarray,
    ) -> float:
        # H(y) = -sum_k p_k log p_k
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(-np.sum(p * np.log(p)))

    def _get_gini_index(
            self,
            y: np.ndarray,
    ) -> float:
        # G(y) = 1 - sum_k p_k^2: the probability that two random draws disagree
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1 - np.sum(p ** 2))

    def _impurity(self, y: np.ndarray) -> float:
        if self.info_criteria == "entropy":
            return self._get_entropy(y)
        elif self.info_criteria == "gini":
            return self._get_gini_index(y)
        raise ValueError(f"Unknown criterion: {self.info_criteria}")

    def _get_decision_criteria(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float]]:
        """Return the (feature, threshold) with the largest information gain,
        or (None, None) if no feature can split these points."""
        # impurity before splitting
        info_before = self._impurity(y)

        best_idx, best_threshold, best_gain = None, None, -np.inf
        # iterate over all features
        for i in range(X.shape[1]):
            # candidate thresholds: midpoints between consecutive unique values
            all_unique_values = np.unique(X[:, i])
            all_thresholds = (all_unique_values[1:] + all_unique_values[:-1]) / 2
            for threshold in all_thresholds:
                left = X[:, i] < threshold
                # information gain = impurity before - weighted impurity after
                information_gain = info_before - (
                    left.mean() * self._impurity(y[left]) +
                    (1 - left.mean()) * self._impurity(y[~left])
                )
                if information_gain > best_gain:
                    best_idx, best_threshold, best_gain = i, threshold, information_gain
        return best_idx, best_threshold

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> None:
        # every node (including leaves) predicts the majority label of its points
        labels, counts = np.unique(y, return_counts=True)
        self.decision = labels[np.argmax(counts)]

        # stopping rules: the node is pure, or the depth budget is used up
        if len(labels) == 1 or self.max_depth == 0:
            return

        # get the decision (splitting) criteria; None if all rows are identical
        category_idx, threshold = self._get_decision_criteria(X, y)
        if category_idx is None:
            return
        self.category_idx = category_idx
        self.threshold = threshold

        # split the data and grow both children (each child is a smaller tree)
        left = X[:, category_idx] < threshold
        child_depth = None if self.max_depth is None else self.max_depth - 1
        self.left = DTNode(self.info_criteria, child_depth)
        self.left.fit(X[left], y[left])
        self.right = DTNode(self.info_criteria, child_depth)
        self.right.fit(X[~left], y[~left])

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        y_pred = []
        for x in X:
            # walk down from the root until we reach a leaf
            node = self
            while not node.is_leaf():
                node = node.left if x[node.category_idx] < node.threshold else node.right
            y_pred.append(node.decision)
        return np.array(y_pred)


class DTRegNode:
    """A node of a regression tree.

    Same greedy algorithm as DTNode, with two changes:
    - impurity is the variance of the targets (so the gain is the reduction in
      mean squared error, as in CART and scikit-learn's "squared_error");
    - a leaf predicts the mean target of its training points.
    """

    def __init__(
            self,
            max_depth: Optional[int] = None,
            min_std: float = 0.01,
    ) -> None:
        self.category_idx: int = -1
        self.threshold: float = -1
        self.left: Optional[DTRegNode] = None
        self.right: Optional[DTRegNode] = None
        self.decision: Optional[float] = None
        self.max_depth = max_depth
        self.min_std = min_std  # stop splitting once the targets are this similar

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def _get_decision_criteria(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float]]:
        # impurity before splitting
        info_before = np.var(y)

        best_idx, best_threshold, best_gain = None, None, -np.inf
        for i in range(X.shape[1]):
            all_unique_values = np.unique(X[:, i])
            all_thresholds = (all_unique_values[1:] + all_unique_values[:-1]) / 2
            for threshold in all_thresholds:
                left = X[:, i] < threshold
                information_gain = info_before - (
                    left.mean() * np.var(y[left]) +
                    (1 - left.mean()) * np.var(y[~left])
                )
                if information_gain > best_gain:
                    best_idx, best_threshold, best_gain = i, threshold, information_gain
        return best_idx, best_threshold

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> None:
        # every node predicts the mean target of its points
        self.decision = float(np.mean(y))

        # stopping rules: targets (almost) constant, or depth budget used up
        if len(y) < 2 or np.std(y) <= self.min_std or self.max_depth == 0:
            return

        category_idx, threshold = self._get_decision_criteria(X, y)
        if category_idx is None:  # all rows identical: cannot split
            return
        self.category_idx = category_idx
        self.threshold = threshold

        left = X[:, category_idx] < threshold
        child_depth = None if self.max_depth is None else self.max_depth - 1
        self.left = DTRegNode(child_depth, self.min_std)
        self.left.fit(X[left], y[left])
        self.right = DTRegNode(child_depth, self.min_std)
        self.right.fit(X[~left], y[~left])

    def predict(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self
            while not node.is_leaf():
                node = node.left if x[node.category_idx] < node.threshold else node.right
            y_pred[i] = node.decision
        return y_pred
