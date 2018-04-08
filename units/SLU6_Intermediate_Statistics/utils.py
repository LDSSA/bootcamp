import pandas as pd
from matplotlib import pyplot as plt
import numpy as np




def get_heights_data_metric():
    return pd.DataFrame({
        'ages': [2, 4, 4, 6, 8, 9, 12, 14],
        'heights': [120, 125, 127, 135, 140, 139, 170, 210]
    })


def get_heights_data_freedom_units():
    return pd.DataFrame({
        'ages': [2, 4, 4, 6, 8, 9, 12, 14],
        'heights': [3.93700787, 4.10104987, 4.16666667, 4.42913386, 4.59317585,
                    4.56036745, 5.57742782, 6.88976378]
    })


def _make_square(df, i):
    x_mean = df.ages.mean()
    y_mean = df.heights.mean()

    x = df.iloc[i].ages
    y = df.iloc[i].heights

    alpha = .1

    if x > x_mean and y > y_mean:
        plt.fill_betweenx(x1=x_mean, x2=x, y=(y_mean, y), alpha=alpha,
                          color='b')

    elif x < x_mean and y < y_mean:
        plt.fill_betweenx(x1=x, x2=x_mean, y=(y, y_mean), alpha=alpha,
                          color='b')

    elif x < x_mean and y > y_mean:
        plt.fill_betweenx(x1=x, x2=x_mean, y=(y_mean, y), alpha=alpha,
                          color='r')

    else:
        plt.fill_betweenx(x1=x_mean, x2=x, y=(y, y_mean), alpha=alpha,
                          color='r')


def quick_scatterplot(df, plot_center=False, plot_squares=None):
    df.plot(kind='scatter', x='ages', y='heights', figsize=(12, 6))

    if plot_center:
        plt.scatter(df.ages.mean(), df.heights.mean(), color='k', marker='+',
                    s=250, )

    if plot_squares:
        if plot_squares == 'all':
            for i in range(len(df)):
                _make_square(df, i)

        else:
            _make_square(df, plot_squares)

    plt.show()


def get_data_for_spearman():
    np.random.seed(100)
    x = np.linspace(-100, 100)
    a = pd.Series(x ** 3)[3::] / 100000
    a.index = a.index * 10
    a = a + a.index / 100
    return a.reset_index().rename(columns={'index': 'a', 0: 'b'})