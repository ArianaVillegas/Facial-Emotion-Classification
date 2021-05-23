from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import get_dataset

X, y = get_dataset('data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# nbrs = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)).fit(X_train, y_train)
nbrs = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0)).fit(X_train, y_train)


titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(nbrs, X_test, y_test,
                                 display_labels=list(set(y)),
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

plt.show()