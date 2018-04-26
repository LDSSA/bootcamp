import numpy as np
import matplotlib.pyplot as plt

def prepare_dataset(df):
    df = df.copy()
    df['before_millenium'] = np.where(df['year'] < 2000, 1, 0)
    df['size_set'] = np.where(df['num_parts'] > 100, 'big set', 'small set')
    return df

def plot_mean_vs_median_hist(df):
    df.num_parts.hist(bins=50, figsize=(17, 8), alpha=0.5)
    plt.axvline(df.num_parts.mean(), color='r')
    plt.axvline(df.num_parts.median(), color='yellow')
    plt.xlabel('Number of parts')
    plt.ylabel('frequency')
    plt.title('Mean vs median')
    plt.show();