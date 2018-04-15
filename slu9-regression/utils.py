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