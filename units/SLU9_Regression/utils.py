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

    
def learning_rate_impact(learning_rate=0.1): 
    f = lambda x: x ** 2
    df_dx = lambda x: 2 * x
    x = -300

    epochs = 100

    data = []

    for epoch in range(epochs):
        x = x - learning_rate * df_dx(x)
        data.append({
            'epoch': epoch, 'x': x, 
            'f(x)': f(x), 'df_dx': df_dx(x)})

    data = pd.DataFrame(data)

    plt.plot(data['epoch'], data['f(x)'])
    plt.ylabel('f(x)')
    plt.xlabel('epochs')
    plt.title('learning rate: {}'.format(learning_rate))

    
def gradient_descent_learning_rate_impact_demo():
    interact(learning_rate_impact, 
             learning_rate=FloatSlider(
                 min=0.001, max=1.1, 
                 step=0.001, value=0.001))
    
    
import numpy as np

import matplotlib.pyplot as plt


def get_computed_function(x, name='convex-1'):
    if name == 'convex-1': 
        return {
            'y': x ** 2, 
            'dy': 2 * x
        }
    if name == 'convex-2':
        return {
            'y': x ** 2 + 3 * x +2, 
            'dy': 2 * x + 3
        }
    if name == 'non-convex-1': 
        return {
            'y': (x ** 2) + abs(15 * x) * np.cos(x), 
            'dy': 2 * x + np.cos(x) * (15 * x / abs(x)) - abs(15 * x) * np.sin(x)
        }
    if name == 'non-convex-2': 
        return {
            'y': 3 * np.cos(x) + abs(x), 
            'dy': -3 * np.sin(x) + x / abs(x)
        }
    if name == 'non-convex-3': 
        w1 = 5.1
        w2 = 10.1
        # TODO: still has problems
        return {
            'dy': 2 * x + w1 * w2 * np.cos(w2 * x) + 2 * (x ** 2) * np.cos(x) * np.sin(x) + 2 * x * (np.sin(x) ** 2) + w1 * np.sin(w1 * x), 
            'y': x ** 2 + (x * np.sin(x)) ** 2 + w1 * np.sin(w2 * x) - np.cos(w1 * x)
        }
    
    
def run_sgd_step(o, learning_rate=0.1, 
                 name='convex-2', 
                 range_def=[-10.0, 10.0, 10000]):
    x_min = min(range_def[0], o['curr_x'])
    x_max = max(range_def[1], o['curr_x'])
    n_steps = range_def[2]
    x = np.linspace(x_min, x_max, n_steps)
    y = get_computed_function(x, name=name)
    y = y['y']
    
    curr_y = get_computed_function(np.array([o['curr_x']]), name=name)
    d_curr_y = curr_y['dy']
    curr_y = curr_y['y']
    
    o['curr_x'] = o['curr_x'] - learning_rate * d_curr_y
    curr_y = get_computed_function(np.array(o['curr_x']), name=name)['y']
    o['curr_x'] = o['curr_x'][0]
    
    plt.plot([o['curr_x']], [curr_y], 'o')
    plt.plot(x, y)
    
    return o


def non_convex_gradient_descent_demo():
    o = {'curr_x': -6.0}

    interact_manual(run_sgd_step, 
                    learning_rate=FloatSlider(min=0.01, max=2.0, step=0.01, value=0.01), 
                    o=fixed(o), 
                    name=fixed('non-convex-1'), 
                    range_def=fixed([-20, 20, 100000]));
    
    

def run_sgd_for_simple_lr(x, y, params, learning_rate):
    run_linear_regression_sgd_epoch(x, y, params, learning_rate)
    
    x_ = np.linspace(-4, 4)
    b0 = params['b0']
    b1 = params['b1']
    y_hat = b0 + b1 * x_
    
    plt.ylim([-4, 8])
    plt.xlim([-6, 6])
    plt.plot(x_, y_hat)
    plt.scatter(x, y, c='orange')
    
    
def run_linear_regression_sgd_epoch(x, y, params, learning_rate): 
    data = np.concatenate((np.array([x]).T, np.array([y]).T), axis=1)
    
    np.random.shuffle(data)
    
    for m in range(x.shape[0]):
        x = data[:, 0]
        y = data[:, 1]
        
        b0 = params['b0']
        b1 = params['b1']

        x_ = x[m]
        y_ = y[m]
        
        y_hat = b0 + b1 * x_

        d_mse_d_b1 = - 2 * (y_ - y_hat) * x_

        d_mse_d_b0 = - 2 * (y_ - y_hat)

        b0 = b0 - learning_rate * d_mse_d_b0
        b1 = b1 - learning_rate * d_mse_d_b1
    
    params['b0'] = b0
    params['b1'] = b1
    
    return params
    

def run_multiple_sgd_iter_for_simple_r(x, y, params, learning_rate, n_iter):
    for i in range(int(n_iter)):
        run_linear_regression_sgd_epoch(x, y, params, learning_rate)
    
        x_ = np.linspace(-4, 4)
        b0 = params['b0']
        b1 = params['b1']
        y_hat = b0 + b1 * x_
    
    run_sgd_for_simple_lr(x, y, params, learning_rate)
    

def sgd_simple_lr_dataset_demo():
    x, y = make_regression(n_features=1, n_samples=100, noise=30.5, random_state=10, bias=200)
    x = x[:, 0]
    y /= 100
    y *= 2.0

    params = {
        'b0': -1, 
        'b1': -5
    }

    interact_manual(run_multiple_sgd_iter_for_simple_r, 
                    x=fixed(x), y=fixed(y), 
                    params=fixed(params), 
                    n_iter=FloatSlider(
                        min=1, max=100, step=1, value=1), 
                    learning_rate=FloatSlider(
                        min=0.01, max=2.0, step=0.01, value=0.01))