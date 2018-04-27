import time
from IPython import display
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

def get_data_wine():
	data = pd.read_csv('data/winequality-red.csv', delimiter=';').drop('quality',axis=1)
	data.columns = data.columns.str.replace(' ', '_')
	return data

def get_wine_quality():
	data = pd.read_csv('data/winequality-red.csv', delimiter=';')['quality']
	return data

def plot_pair_plots(data):
	sns.pairplot(data)

def compute_distances(origin, points, metric="euclidean"):
	origin = origin.values.reshape(1, -1)
	distances = []
	for point in points.values:
		if metric=="euclidean":
			distances.append(pairwise_distances(origin, point.reshape(1, -1), metric)[0][0])
		if metric=="cosine":
			distances.append(pairwise_distances(origin, point.reshape(1, -1), metric)[0][0])
		if metric=="manhattan":
			distances.append(pairwise_distances(origin, point.reshape(1, -1), metric)[0][0])
	return distances

def plot_density_chlorides(data):
	f, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(data['density'], data['chlorides'], s=50,cmap="RdBu")
	ax.set(xlabel="density", ylabel="chlorides", title="Pairwise plot of normalized density and chlorides")
	plt.show()

def plot_kmeans_density_chlorides(data, random_state = 123, k=2):
	new_data = data[['density','chlorides']].values

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = new_data[:, 0].min() - 1, new_data[:, 0].max() + 1
	y_min, y_max = new_data[:, 1].min() - 1, new_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	f, ax = plt.subplots(figsize=(8, 6))
	for i in range(10):
		kmeans = KMeans(init='random', n_clusters=k, n_init=1, max_iter=i+1, random_state=random_state)
		kmeans.fit(new_data)
		Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		display.clear_output(wait=True)
		time.sleep(0.1)
		centroids = kmeans.cluster_centers_
		contour = ax.contourf(xx, yy, Z, 25, cmap="RdBu")
		preds = kmeans.predict(new_data)
		ax.scatter(new_data[:, 0], new_data[:, 1],s=50,c=preds)
		ax.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', cmap="RdBu", s=169, c="white", linewidths=3)
		ax.set(xlabel="density", ylabel="chlorides", title="K-means clustering of normalized density and chlorides (iter %0.1i)" % (i+1))
		display.display(plt.gcf());
	plt.clf();

def plot_toy_dataset(n_samples=1500, random_state=8):
	blobs = datasets.make_blobs(n_samples=1500, random_state=8)[0]
	f, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(blobs[:, 0], blobs[:, 1],s=50)
	plt.show();
	return blobs

def plot_elbow_method(data):
	# k means determine k
	distortions = []
	K = range(1,10)
	for k in K:
	    kmeans = KMeans(n_clusters=k).fit(data)
	    kmeans.fit(data)
	    distortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
	# Plot the elbow
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('K')
	plt.ylabel('Average within cluster distance to centroids')
	plt.title('The Elbow Method showing the optimal k')
	plt.show();

def plot_kmeans(data, random_state = 123, k=3):
	sclr = MinMaxScaler().fit(data)
	new_data = sclr.transform(data)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = new_data[:, 0].min() - 1, new_data[:, 0].max() + 1
	y_min, y_max = new_data[:, 1].min() - 1, new_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	f, ax = plt.subplots(figsize=(8, 6))
	for i in range(10):
		kmeans = KMeans(init='random', n_clusters=k, n_init=1, max_iter=i+1, random_state=random_state)
		kmeans.fit(new_data)
		Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		display.clear_output(wait=True)
		time.sleep(0.1)
		centroids = kmeans.cluster_centers_
		contour = ax.contourf(xx, yy, Z, 25, cmap="RdBu")
		preds = kmeans.predict(new_data)
		ax.scatter(new_data[:, 0], new_data[:, 1],s=50,c=preds)
		ax.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', cmap="RdBu", s=169, c="white", linewidths=3)
		ax.set(title="K-means clustering (iter %0.1i)" % (i+1))
		display.display(plt.gcf());
	plt.clf();

def plot_cumulative_explained_variance_ratio(data):
	new_data = data.copy()
	pca = PCA().fit(new_data)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance');

def plot_two_dimensions(data, label):
	data["quality"] = label.map(str).copy()
	sns.lmplot(data=data, x="first_component", y="second_component", hue="quality", fit_reg=False, scatter_kws={'alpha':0.6})