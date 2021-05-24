from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.experiments import Experiment


# Classifiers
svm = LinearSVC()
knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
dt = DecisionTreeClassifier(random_state=0)

# Experiments
experiment = Experiment('faces', 'results/faces')

# Add classifiers
experiment.addClassifier((svm, 'svm'))
experiment.addClassifier((knn, 'knn'))
experiment.addClassifier((dt, 'dt'))

# Confusion matrix
experiment.crossValidation()

