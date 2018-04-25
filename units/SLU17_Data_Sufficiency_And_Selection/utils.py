import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

categoricals = [
    'PassengerId',
    'Name',
    'Ticket',
    'Sex',
    'Cabin',
    'Embarked',
]

# Here are a few functions that we will use to do our proof of concepts
# around training and testing for kaggle submissions for the titanic
# competition.

# The reason that we need to have these functions they way they are
# is to manage the preprocessing of the features the same for the 
# training and test set. Don't worry too much about exactly what's
# going on in these functions right now but rather focus on the concepts
# that are being covered after this in the notebook.


def read_and_get_dummies(drop_columns=[]):
    # when working inside of functions, always call your dataframe
    # _df so that you know you're never using any from the outside!
    _df = pd.read_csv('data/titanic.csv')

    # now drop any columns that are specified as needing to be dropped
    for colname in drop_columns:
        _df = _df.drop(colname, axis=1)
    
    for colname in categoricals:
        if colname in drop_columns:
            continue
        _df[colname] = _df[colname].fillna('null').astype('category')


    # Split the factors and the target
    X, y = _df.drop('Survived', axis=1), _df['Survived']

    # take special note of this call!
    X = pd.get_dummies(X, dummy_na=True).fillna(-1)

    return _df, X, y
    


def train_and_test(drop_columns=[], max_depth=None, test_size=0.2):
    """
    Train a decision tree and return the classifier, the X_train,
    and the original dataframe so that they can be used on the test
    set later on.
    """
    
    _df, X, y = read_and_get_dummies(drop_columns=drop_columns)
    
    # Now let's get our train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=test_size)
    
    X_train, X_test, y_train, y_test
    
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    clf.fit(X_train, y_train)
    
    score = accuracy_score(y_test, clf.predict(X_test))
    print('X_test accuracy {}'.format(score))
    print('X_train shape: {}'.format(X_train.shape))
    
    return X_train, _df, clf


def produce_test_predictions(train_df, clf, drop_columns=[]):
    _df = pd.read_csv('data/titanic-test.csv')
    
    for colname in drop_columns:
        _df = _df.drop(colname, axis=1)
        
    for colname in categoricals:
        if colname in drop_columns:
            continue
        _df[colname] = _df[colname].fillna('null')
        _df[colname] = pd.Categorical(
            _df[colname],
            categories=train_df[colname].cat.categories
        )
        
    X = pd.get_dummies(_df, dummy_na=True).fillna(-1)
    
    return pd.DataFrame({
        'PassengerId': pd.read_csv('data/titanic-test.csv').PassengerId,
        'Survived': pd.Series(clf.predict(X))
    })


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# from sklearn import tree
# tree.export_graphviz(clf, out_file='tree.dot', feature_names=X_train.columns)
# ! dot -Tpng tree.dot -o tree.png