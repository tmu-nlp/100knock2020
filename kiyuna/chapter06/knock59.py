"""
59. ハイパーパラメータの探索
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．

[Ref]
- https://scikit-learn.org/stable/modules/multiclass.html

[MEMO]
New
"""
import os
import sys
from collections import defaultdict

import timeout_decorator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from knock53 import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import dump, load  # noqa: E402 isort:skip

datasets = ["train", "valid", "test"]


if __name__ == "__main__":
    data = {}
    for dataset in datasets:
        data[dataset] = load_dataset(f"./{dataset}.feature.txt")

    def get_data(dataset, need_dense):
        features, labels = data[dataset]
        if need_dense:
            return features.toarray(), labels
        return features, labels

    clfs = [
        (LinearDiscriminantAnalysis(), True, "Too slow"),
        (QuadraticDiscriminantAnalysis(), True, "Too slow"),
        (ExtraTreesClassifier(), False),
        (RandomForestClassifier(), False),
        (LogisticRegression(multi_class="multinomial", solver="lbfgs"), False),
        (LogisticRegressionCV(multi_class="multinomial"), False, "今はコメントアウト"),
        (RidgeClassifier(), False),
        (RidgeClassifierCV(), False, "Too slow"),
        (BernoulliNB(), False),
        (GaussianNB(), True, "Slow"),
        (KNeighborsClassifier(), False),
        (NearestCentroid(), False),
        (RadiusNeighborsClassifier(), False, "ValueError"),
        (MLPClassifier(), False, "Too slow"),
        (LabelPropagation(), True, "Too slow"),
        (LabelSpreading(), True, "Too slow"),
        (LinearSVC(multi_class="crammer_singer"), False),
        (DecisionTreeClassifier(), False),
        (ExtraTreeClassifier(), False),
    ]
    # clfs = []
    # param_grid = {
    #     "penalty": ["l1", "l2", "lasticnet", "none"],
    #     "dual": [False, True],
    #     "tol": [1e-3, 1e-4, 1e-5],
    #     "C": [10, 1, 0.1],
    #     "fit_intercept": [False, True],
    #     "class_weight": [None, "balanced"],
    #     "solver": ["newton-cg", "sag", "saga", "lbfgs"],
    #     "multi_class": ["multinomial"],
    #     "warm_start": [False, True],
    # }
    # for params in ParameterGrid(param_grid):
    #     clfs.append((LogisticRegression(**params), False))

    @timeout_decorator.timeout(3)
    def clf_fit(clf):
        clf.fit(*get_data("train", need_dense))

    models = defaultdict(list)
    for i, (clf, need_dense, *args) in enumerate(clfs):
        message(i)
        message(type(clf).__name__, type="status")
        message(clf.get_params())
        if args:
            message("skip", args, type="warning")
            continue
        if (
            clf.get_params().get("penalty", None) == "l1"
            and clf.get_params().get("solver", None) == "saga"
        ):
            message("skip", "Too slow", type="warning")
            continue
        try:
            clf_fit(clf)
            score = clf.score(*get_data("valid", need_dense))
            models[score].append(clf)
            message(score, type="success")
        except Exception as e:
            message("skip", e, type="warning")
    best_score = max(models)
    message(f"=== best_score (valid) {best_score:f} ===", type="success")
    for best_model in models[best_score]:
        message(type(best_model).__name__, type="status")
        message(
            f"score (test) {best_model.score(*get_data('test', need_dense)):f}", type="success"
        )

        """
        いろんな Classifier を試した結果
        [+] === best_score (valid) 0.921348 ===
        [x] RidgeClassifier
        [+] score (test) 0.928090
        [x] LinearSVC
        [+] score (test) 0.929588

        LogisticRegression(**params) を頑張った結果
        [+] === best_score (valid) 0.919101 ===
        [x] LogisticRegression
        [+] score (test) 0.926592
        [x] LogisticRegression
        [+] score (test) 0.916854
        """
