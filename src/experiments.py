import os

import numpy as np

from sklearn.model_selection import train_test_split, KFold

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.plots import plot_error, plot_confusion_matrix
from src.utils import get_dataset, Bootstrap_split, make_dir

np.random.seed(2021)
np.set_printoptions(suppress=True)


class Experiment:
    def __init__(self, input_path, output_path):
        self.X, self.y = get_dataset(input_path)
        self.label = list(set(self.y))
        self.output_path = output_path
        self.classifiers = []

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def addClassifier(self, classifier):
        clf = make_pipeline(StandardScaler(), classifier[0])
        self.classifiers.append((classifier[1], clf))

    def error(self, groups, classifier):
        error = []
        cm = np.zeros((len(self.label), len(self.label)))
        cnt = 0
        for train, test in groups:
            classifier.fit([self.X[i] for i in train], [self.y[i] for i in train])
            y_p = classifier.predict([self.X[i] for i in test])
            y_test = [self.y[i] for i in test]
            error.append(sum(y_p != y_test) / len(y_test))
            for i in range(len(y_test)):
                cm[self.label.index(y_p[i])][self.label.index(y_test[i])] += 1
            cnt += 1
        cm = np.round(cm/cnt).astype(int)
        return np.array(error), cm

    def crossValidation(self, n_splits=10):
        new_path = make_dir(self.output_path + "/cross_validation")
        k_fold = make_dir(new_path + "/k_fold")
        bootstrap = make_dir(new_path + "/bootstrap")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        bt_train, bt_test = Bootstrap_split(self.X, n_splits=n_splits)

        for classifier in self.classifiers:
            for method, path in zip([kf.split(self.X), zip(bt_train, bt_test)], [k_fold, bootstrap]):
                error, cm = self.error(method, classifier[1])
                plot_error(error, path, classifier[0])
                plot_confusion_matrix(classifier[0], path, cm, self.label)
                np.savetxt(path + '/confusion_matrix_'+classifier[0]+'.txt', cm, delimiter=',')

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
