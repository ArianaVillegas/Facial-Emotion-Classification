import numpy as np
from scipy import stats
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from utils import get_dataset

X, y = get_dataset('leedsbutterfly/images')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = make_pipeline(StandardScaler(), LinearSVC())
# clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights='distance'))
# clf = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
# clf.fit(X_train, y_train)


# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(clf, X_test, y_test,
#                                  display_labels=list(set(y_test)),
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#
# plt.show()


def plot_norm(mu, sigma, name):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=name)


cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf, X, y, scoring='accuracy', cv=cv, return_train_score=True, n_jobs=-1)
plot_norm(scores['test_score'].mean(), scores['test_score'].std(), 'test')
plot_norm(scores['train_score'].mean(), scores['train_score'].std()+0.001, 'train')
plt.legend()
plt.show()

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
