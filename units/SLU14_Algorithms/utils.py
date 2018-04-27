import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances
from sklearn.datasets import make_moons, make_circles, make_blobs, make_checkerboard
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm = plt.cm.jet
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def classifier_decision_graphs(classifier, datasets):
    i=1
    for ds in datasets:
        figure = plt.figure(figsize=(20, 15))

        h = 0.02
        
        X, y = ds

        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # just plot the dataset first
        cm = plt.cm.jet
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), 5, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.7)
        # and testing points
        #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        clf = classifier
        clf.fit(X_train, y_train)

        clf_predictions = clf.predict_proba(X_test)

        score = roc_auc_score(y_test, clf_predictions[:,1])

        #######
        
        i += 1
        
        ax = plt.subplot(len(datasets), 5, i)
        
        Z1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z1 = Z1.reshape(xx.shape)
        ax.contourf(xx, yy, Z1, cmap=cm_bright, alpha=.3)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.6, linewidths=0.6, edgecolors="white")
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

    plt.show()


def classifier_decision_graph(classifier, X, y):
    figure = plt.figure()

    h = 0.02

    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = classifier
    clf.fit(X_train, y_train)

    clf_predictions = clf.predict_proba(X_test)

    score = roc_auc_score(y_test, clf_predictions[:,1])

    print('ROC AUC Score:')
    print(score)

    Z1 = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z1 = Z1.reshape(xx.shape)
    plt.contourf(xx, yy, Z1, cmap=cm_bright, alpha=.3)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.6, linewidths=0.6, edgecolors="white")
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    plt.show()