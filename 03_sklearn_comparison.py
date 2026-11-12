"""03: the same trees with scikit-learn, as a sanity check.

What is NEW compared to 01 and 02
- scikit-learn's DecisionTreeClassifier / DecisionTreeRegressor implement the
  same greedy CART algorithm (in optimised C code). With the same criterion
  and max_depth, their scores should be close to ours. Small differences come
  from tie-breaking between equally good splits.
- From here on (see the Ensemble-Methods repository) we can use the fast
  library trees as building blocks.

Run: python 03_sklearn_comparison.py --task clf
     python 03_sklearn_comparison.py --task reg
"""
import argparse

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import load_data, standardize_targets, rmse
from model import DTNode, DTRegNode


def compare_classification(args) -> None:
    X_train, X_test, y_train, y_test = load_data(args.car_path)
    ours = DTNode(args.criteria, args.max_depth)
    theirs = DecisionTreeClassifier(criterion=args.criteria, max_depth=args.max_depth)
    for name, model in [("Ours", ours), ("Sklearn", theirs)]:
        model.fit(X_train, y_train)
        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (model.predict(X_test) == y_test).mean()
        print(f"{name:8s} train accuracy: {train_acc:.4f} | test accuracy: {test_acc:.4f}")


def compare_regression(args) -> None:
    X_train, X_test, y_train, y_test = load_data(args.house_path, is_regression=True)
    y_train, y_test = standardize_targets(y_train, y_test)
    ours = DTRegNode(max_depth=args.max_depth)
    theirs = DecisionTreeRegressor(criterion="squared_error", max_depth=args.max_depth)
    for name, model in [("Ours", ours), ("Sklearn", theirs)]:
        model.fit(X_train, y_train)
        print(f"{name:8s} train RMSE: {rmse(model.predict(X_train), y_train):.4f} | "
              f"test RMSE: {rmse(model.predict(X_test), y_test):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["clf", "reg"], default="clf")
    parser.add_argument("--car_path", default="data/car.csv")
    parser.add_argument("--house_path", default="data/house.csv")
    parser.add_argument("--criteria", choices=["entropy", "gini"], default="entropy")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="maximum number of splits from root to leaf (-1: unlimited)")
    args = parser.parse_args()
    if args.max_depth < 0:
        args.max_depth = None  # unlimited depth, for both implementations
    if args.task == "clf":
        compare_classification(args)
    else:
        compare_regression(args)
