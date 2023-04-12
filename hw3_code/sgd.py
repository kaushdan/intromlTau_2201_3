#################################
# Your name: Daniel Kaushansky
#################################
import numpy as np
import numpy.random
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.datasets import fetch_openml

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml("mnist_784", as_frame=False)
    data = mnist["data"]
    labels = mnist["target"]

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0]
    )
    test_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0]
    )

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(
        validation_data_unscaled, axis=0, with_std=False
    )
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(data[0].shape)

    for t in range(1, T + 1):
        eta_t = eta_0 / t
        idx = np.random.randint(0, high=data.shape[0], dtype=int)

        w = w * float((1 - eta_t))
        if labels[idx] * np.dot(w, data[idx]) < 1:
            w = w + data[idx] * float(eta_t * C * labels[idx])

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(data[0].shape)

    for t in range(1, T + 1):
        eta_t = eta_0 / t
        idx = np.random.randint(0, high=data.shape[0], dtype=int)
        w = np.subtract(w, eta_t * log_loss_gradient(w, data[idx], labels[idx]))

    return w


#################################

# Place for additional code


def log_loss_gradient(w, sample, label):
    return np.dot(softmax(-1 * label * np.dot(w, sample)), sample)


def zo_loss(w, sample, label):
    return 0 if label * np.dot(w, sample) > 0 else 1


def summed_lost(lost_func, w, data, labels, **kwargs):
    return np.sum(
        [lost_func(w, sample, label, **kwargs) for sample, label in zip(data, labels)]
    ) / len(data)


def calc_accuracy(lost_func, w, data, labels, **kwargs):
    return 1 - summed_lost(lost_func, w, data, labels, **kwargs)


def cross_validation(ws, lost_func, validation_data, validation_labels, **kwargs):

    accuracies = np.array(
        [
            calc_accuracy(lost_func, w, validation_data, validation_labels, **kwargs)
            for w in ws
        ]
    )

    return accuracies


def section_a(q=1, T=1000, C=1):
    print(f"in q_{q}_section_a")
    etas = np.power(10, np.arange(-5, 4), dtype=float)
    train_data, train_labels, validation_data, validation_labels, *_ = helper()
    avg_accuracies = []
    for eta_0 in etas:
        if q == 1:
            ws = [
                SGD_hinge(data=train_data, labels=train_labels, C=C, eta_0=eta_0, T=T)
                for _ in range(10)
            ]

        else:
            ws = [
                SGD_log(data=train_data, labels=train_labels, eta_0=eta_0, T=T)
                for _ in range(10)
            ]

        accuracies = cross_validation(
            ws,
            zo_loss,
            validation_data,
            validation_labels,
        )

        avg_accuracies.append(np.average(accuracies))

    plt.semilogx(etas, avg_accuracies, "o-", color="g", label="average accuracy")
    plt.axis([10 ** (-6), 10**4, 0, 1])
    plt.grid(True)
    plt.title(
        f"Accuracies on validation set with zo-loss minimizing {'hinge log with l2 regularization' if q == 1 else 'log loss'} SGD"
    )
    plt.xlabel("eta")
    plt.ylabel("accuracies")
    plt.legend()
    plt.show()
    print(f"best eta: {etas[np.argmax(avg_accuracies)]}")


def section_b(q=1, T=1000, eta=1):
    print(f"in q_{q}_section_b")
    cs = np.power(10, np.arange(-5, 6), dtype=float)
    train_data, train_labels, validation_data, validation_labels, *_ = helper()
    avg_accuracies = []
    for c in cs:
        if q == 1:
            ws = [
                SGD_hinge(data=train_data, labels=train_labels, C=c, eta_0=eta, T=T)
                for _ in range(10)
            ]

        else:
            ws = [
                SGD_log(data=train_data, labels=train_labels, eta_0=eta, T=T)
                for _ in range(10)
            ]

        accuracies = cross_validation(
            ws,
            zo_loss,
            validation_data,
            validation_labels,
        )

        avg_accuracies.append(np.average(accuracies))

    plt.semilogx(cs, avg_accuracies, "o-", color="g", label="average accuracy")
    plt.axis([10 ** (-6), 10**6, 0.95, 1])
    plt.grid(True)
    plt.xlabel("C")
    plt.ylabel("accuracies")
    plt.ylabel("accuracies")
    plt.legend()
    plt.title(
        "Accuracies on validation set with zo-loss minimizing hinge and l2 regularization SGD"
    )
    plt.show()
    print(f"best c: {cs[np.argmax(avg_accuracies)]}")


def section_c(q=1, T=20000, eta=1, C=0.0001):
    print(f"in q_{q}_section_c")
    (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    ) = helper()
    if q == 1:
        ws = [
            SGD_hinge(data=train_data, labels=train_labels, C=C, eta_0=eta, T=T)
            for _ in range(10)
        ]

    else:
        ws = [
            SGD_log(data=train_data, labels=train_labels, eta_0=eta, T=T)
            for _ in range(10)
        ]

    accuracies = cross_validation(
        ws,
        zo_loss,
        validation_data,
        validation_labels,
    )

    avg_accuracies = np.average(accuracies)

    w = ws[np.argmax(avg_accuracies)]

    plt.imshow(w.reshape((28, 28)), interpolation="nearest")
    plt.show()

    print(f"accuracy is: {calc_accuracy(zo_loss, w, test_data, test_labels)}")


def q2_section_c():
    T = 20000
    eta = 100
    train_data, train_labels, *_ = helper()

    w = np.zeros(train_data[0].shape)
    norms = []
    ranges = np.arange(1, T + 1)

    for t in ranges:
        eta_t = eta / t
        idx = np.random.randint(0, high=train_data.shape[0], dtype=int)
        w = np.subtract(
            w, eta_t * log_loss_gradient(w, train_data[idx], train_labels[idx])
        )
        norms.append(np.dot(w, w))

    plt.plot(ranges, norms, "r-")
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel("norm")
    plt.show()


#################################

if __name__ == "__main__":
    section_c(q=2, eta=0.1)
