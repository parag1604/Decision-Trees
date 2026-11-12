"""01: a classification tree from scratch (Car Evaluation dataset).

What this file teaches
- How a decision tree is grown greedily: at each node, try every feature and
  every threshold, and keep the split with the largest information gain
      gain = impurity(parent) - sum_child (n_child / n) * impurity(child)
- Two impurity measures: entropy (-sum p log p) and the Gini index
  (1 - sum p^2). Compare them with --criteria.
- max_depth limits how many questions the tree may ask. Compare train and
  test accuracy for small and large depths to see under- and over-fitting.

Run: python 01_classification_tree.py --criteria entropy --max_depth 5
"""
import argparse

from utils import load_data
from model import DTNode as DTModel


def main(args) -> None:
    # load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)

    # train model
    model = DTModel(args.criteria, args.max_depth)
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_train)
    acc = (y_pred == y_train).mean()
    print(f"Train accuracy: {acc:.4f}")

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_path", default="data/car.csv")
    parser.add_argument("--criteria", choices=["entropy", "gini"], default="entropy")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="maximum number of splits from root to leaf (-1: unlimited)")
    main(parser.parse_args())
