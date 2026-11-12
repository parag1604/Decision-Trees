"""02: a regression tree from scratch (house prices dataset).

What is NEW compared to 01
- The target is a real number (price), so "impurity" becomes the variance of
  the targets in a node, and the gain is the reduction in squared error.
- A leaf predicts the mean target of its training points, so the tree is a
  piecewise-constant function of the features.
- Prices are standardised with the training mean/std, so an RMSE of 1.0 is
  exactly as good as always predicting the mean price.

Run: python 02_regression_tree.py --max_depth 5
"""
import argparse

from utils import load_data, standardize_targets, rmse
from model import DTRegNode as DTRegModel


def main(args) -> None:
    # load data
    X_train, X_test, y_train, y_test = load_data(
        args.data_path, is_regression=True)

    # normalize the targets
    y_train, y_test = standardize_targets(y_train, y_test)

    # train model
    model = DTRegModel(max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # predict
    print(f"Train RMSE: {rmse(model.predict(X_train), y_train):.4f}")
    print(f"Test RMSE: {rmse(model.predict(X_test), y_test):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_path", default="data/house.csv")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="maximum number of splits from root to leaf (-1: unlimited)")
    main(parser.parse_args())
