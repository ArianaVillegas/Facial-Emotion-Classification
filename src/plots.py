import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def plot_norm(mu, sigma, name):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=name)


def plot_variance(scores, new_path, classifier):
    plot_norm(scores['test_score'].mean(), scores['test_score'].std(), 'test')
    plot_norm(scores['train_score'].mean(), scores['train_score'].std() + 0.001, 'train')
    plt.legend()
    plt.savefig(new_path + "/" + classifier + ".png")
    plt.close()


def plot_error(error, new_path, classifier):
    plot_norm(error.mean(), error.std(), 'error')
    plt.savefig(new_path + "/" + classifier + ".png")
    plt.close()
