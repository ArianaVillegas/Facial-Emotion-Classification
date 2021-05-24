import os

import numpy as np
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split, KFold

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.plots import plot_error
from src.utils import get_dataset, Bootstrap_split

np.random.seed(2021)


class Experiment:
    def __init__(self, input_path, output_path):
        self.X, self.y = get_dataset(input_path)
        self.output_path = output_path
        self.classifiers = []

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def addClassifier(self, classifier):
        clf = make_pipeline(StandardScaler(), classifier[0])
        self.classifiers.append((classifier[1], clf))

    def error(self, groups, classifier):
        error = []
        for train, test in groups:
            classifier.fit([self.X[i] for i in train], [self.y[i] for i in train])
            y_p = classifier.predict([self.X[i] for i in test])
            error.append(sum(y_p != [self.y[i] for i in test]) / len(y_p))
        return np.array(error)

    def crossValidation(self, n_splits=10):
        new_path = self.output_path + "/cross_validation"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        k_fold = "/k_fold"
        if not os.path.exists(new_path + k_fold):
            os.makedirs(new_path + k_fold)
        bootstrap = "/bootstrap"
        if not os.path.exists(new_path + bootstrap):
            os.makedirs(new_path + bootstrap)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        bt_train, bt_test = Bootstrap_split(self.X, n_splits=n_splits)

        for classifier in self.classifiers:
            error_kf = self.error(kf.split(self.X), classifier[1])
            plot_error(error_kf, new_path + k_fold, classifier[0])

            error_bt = self.error(zip(bt_train, bt_test), classifier[1])
            plot_error(error_bt, new_path + bootstrap, classifier[0])

    def getConfusionMatrix(self, test_size=0.3):
        new_path = self.output_path + "/confusion_matrix"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        for classifier in self.classifiers:
            classifier[1].fit(X_train, y_train)
            titles_options = [("Confusion matrix, without normalization", None),
                              ("Normalized confusion matrix", 'true')]
            for title, normalize in titles_options:
                if not os.path.exists(new_path + "/" + title):
                    os.makedirs(new_path + "/" + title)

                disp = plot_confusion_matrix(classifier[1], X_test, y_test,
                                             display_labels=list(set(self.y)),
                                             cmap=plt.cm.Blues,
                                             normalize=normalize)
                disp.ax_.set_title(title)

                plt.savefig(new_path + "/" + title + "/plot_confusion_matrix_" + classifier[0] + ".png")
