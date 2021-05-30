import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sn


def plot_norm(mu, sigma, name):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=name)


def plot_variance(scores, new_path, classifier):
    plot_norm(scores['test_score'].mean(), scores['test_score'].std(), 'test')
    plot_norm(scores['train_score'].mean(), scores['train_score'].std(), 'train')
    plt.legend()
    plt.savefig(new_path + "/" + classifier + ".png")
    plt.close()


def plot_error(error, new_path, classifier):
    plot_norm(error.mean(), error.std() + 0.0001, 'error')
    plt.savefig(new_path + "/" + classifier + ".png")
    plt.close()


def plot_confusion_matrix(classifier, path, cm, label):
    titles_options = [("Confusion matrix, without normalization", False),
                      ("Normalized confusion matrix", True)]
    df_cm = pd.DataFrame(cm, label, label)
    for title, normalize in titles_options:
        if normalize:
            df_cm = df_cm.apply(lambda x: x/np.sum(x), axis=1)
        ax = plt.axes()
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="YlGnBu")
        ax.set_title(title)

        norm = "normalized_" if normalize else ""
        plt.savefig(path + "/confusion_matrix_" + norm + classifier + ".png")
        plt.close()
