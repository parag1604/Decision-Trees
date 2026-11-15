# Decision Trees

Code for classification and regression trees (CART) written from scratch in numpy, and checked against scikit-learn.

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Decision-Trees-and-Ensembles) (the sections on classification and regression trees, impurity and pruning).

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy - required for vectorized operations
- Pandas - for reading and encoding the CSV files
- Scikit-learn - for the reference trees in `03_sklearn_comparison.py`

## Data

Both datasets are small and are **included in this repository** under `./data` (pass other paths with `--data_path`, `--car_path` or `--house_path`):

```
data/
  car.csv     # classification
  house.csv   # regression
```

**car.csv**: the [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation) dataset (1728 cars, 6 categorical features). It is included as `data/car.csv` (with a header row; `5more` doors is written `5`). If you use the original `car.data` instead, a header row is optional; the columns are read in the UCI order: buying price, maintenance cost, doors, number of persons, lug_boot, safety, decision. The target is made binary: acceptable (`acc`, `good`, `vgood`) vs. unacceptable (`unacc`). The ordered categories (e.g. low < med < high < vhigh) are encoded as ordered numbers.

**house.csv**: the Kaggle [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) (`Housing.csv`, 545 houses), included as `data/house.csv`. The code expects a header row with these columns:

| column | type |
|---|---|
| `price` | target |
| `area`, `bedrooms`, `bathrooms`, `stories`, `parking` | numbers |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | `yes` / `no` |
| `furnishingstatus` | `furnished` / `semi-furnished` / `unfurnished` |

Both files are split 80/20 into train and test sets with a fixed seed.

## Running

### 1. Classification tree

```bash
python 01_classification_tree.py --criteria entropy --max_depth 5
```

Grows a binary tree greedily: at every node it tries every feature and every threshold and keeps the split with the largest information gain. Compare `--criteria entropy` with `--criteria gini`, and compare the train/test accuracy for small (`--max_depth 1`) and large (`--max_depth -1`, unlimited) trees to see under- and over-fitting.

### 2. Regression tree

```bash
python 02_regression_tree.py --max_depth 5
```

New: the target is a price, so the impurity of a node is the variance of its targets and each leaf predicts the mean price of its training points. Prices are standardized with the training statistics, so a test RMSE of 1.0 is no better than always predicting the mean price.

### 3. Comparison with scikit-learn

```bash
python 03_sklearn_comparison.py --task clf
python 03_sklearn_comparison.py --task reg
```

New: the same experiments with `DecisionTreeClassifier` / `DecisionTreeRegressor`. With the same criterion and depth the scores should match ours (up to tie-breaking between equally good splits). This is the only script that uses scikit-learn; `model.py` and scripts 01–02 are from scratch.

**Results on the included data** (MacBook CPU, default settings): the classification tree reaches 93.4% train / 91.3% test accuracy on car.csv, identical to scikit-learn's `DecisionTreeClassifier` with the same criterion and depth; the regression tree reaches a standardized RMSE of 0.49 train / 0.77 test on house.csv.

As an additional check, `model.py` was compared with scikit-learn on its built-in breast-cancer (classification, first 10 features; entropy and Gini, depths 3, 6 and unlimited) and diabetes (regression with `min_std=0`, depths 3 and 6) datasets: the training-set predictions agree on 100% of the points for every setting. The [Ensemble-Methods](https://github.com/parag1604/Ensemble-Methods) repository builds bagging, random forests and boosting on a from-scratch copy of these trees (with a vectorised split search, feature subsampling and sample weights).
