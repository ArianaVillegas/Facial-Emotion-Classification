from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.experiments import Experiment


# Classifiers
svm = LinearSVC(dual=True, max_iter=30000)
knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
dt = DecisionTreeClassifier(random_state=0)

# Experiments
experiment = Experiment('faces', 'results/faces')

# Add classifiers
experiment.addClassifier((knn, 'knn'))
experiment.addClassifier((svm, 'svm'))
experiment.addClassifier((dt, 'dt'))

# Croos Validation
experiment.crossValidation()

