from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import FloatSlider, Dropdown
import ipywidgets as widgets

import pandas as pd

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import numpy as np


def plot_simple_regression(b0=0, b1=1, xlim=(-5, 5), ylim=(-5, 5)):
    x = np.linspace(-10, 10, 1000)
    y = b0 + b1 * x
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(x, y)
    plt.plot([0, 0], ylim, 'g-', 
             xlim, [0, 0], 'g-', linewidth=0.4)


def simple_linear_regression_manual_demo_1(): 
    interact(plot_simple_regression, 
         b0=FloatSlider(min=-10, max=10, step=0.01, value=0), 
         b1=FloatSlider(min=-10, max=10, step=0.01, value=1), 
         xlim=fixed((-5, 5)), 
         ylim=fixed((-5, 5)));


def plot_simple_regression_with_dataset(x, y, b0=0, b1=1, xlim=(-5, 5), ylim=(-5, 5)):
    plot_simple_regression(b0, b1, xlim, ylim)
    plt.scatter(x, y)
    
    y_hat = b0 + b1 * x
    
    return "Mean Squared Error (MSE): {}".format(mean_squared_error(y, y_hat))


def simple_linear_regression_demo_2():
    x, y = make_regression(n_features=1, n_samples=100, noise=30.5, random_state=10, bias=200)
    x = x[:, 0]
    y /= 100
    y *= 2.0

    interact(plot_simple_regression_with_dataset, 
             b0=FloatSlider(min=-10, max=10, step=0.01, value=-1), 
             b1=FloatSlider(min=-10, max=10, step=0.01, value=-1), 
             x=fixed(x), 
             y=fixed(y), 
             xlim=fixed((-5, 5)), 
             ylim=fixed((-3, 8)));


def gradient_descent_learning_rate_impact_demo():
    f = lambda x: x ** 2
    df_dx = lambda x: 2 * x
    x = -300

    epochs = 100
    learning_rate = 0.1

    data = []

    for epoch in range(epochs):
        x = x - learning_rate * df_dx(x)
        data.append({
            'epoch': epoch, 'x': x, 
            'f(x)': f(x), 'df_dx': dy_dx(x)})

    data = pd.DataFrame(data)

    plt.plot(data['epoch'], data['f(x)'])
    plt.ylabel('f(x)')
    plt.xlabel('epochs')
    plt.title('learning rate: {}'.format(learning_rate))