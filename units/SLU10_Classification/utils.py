import time
from IPython import display
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

def get_data_iris():
	X, Y = load_iris(True)
	X = pd.DataFrame(X, columns = ['SEPAL_LENGTH', 'SEPAL_WIDTH', 'PETAL_LENGTH', 'PETAL_WIDTH'])
	Y = pd.Series(Y, name = 'SPECIES')
	return X, Y

def plot_pair_plots(X, Y):
	data = pd.concat((X,Y),axis=1)
	sns.pairplot(data,hue="SPECIES")

def super_simple_classifier_plot(X, Y):
	data = pd.concat((X,Y),axis=1)
	data.loc[data["SPECIES"]!=0, "SPECIES"] = 1
	sns.lmplot("SEPAL_LENGTH", "PETAL_WIDTH",data=data, hue="SPECIES", fit_reg=False)
	plt.plot(np.linspace(4,8,1000), [0.8]*1000, 'r-')

def linear_separation_plot(X, Y):
	data = pd.concat((X,Y),axis=1)
	xx, yy = np.mgrid[4:8:.01, 0:2.8:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	for species in [2,1,0]:
		labels = data.loc[data.SPECIES != species,'SPECIES'].unique()
		clf = LogisticRegression(random_state=0).fit(data.loc[data.SPECIES != species,['SEPAL_LENGTH','PETAL_WIDTH']], 
			data.loc[data.SPECIES != species,'SPECIES'])
		probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
		f, ax = plt.subplots(figsize=(8, 6))
		contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
		                      vmin=0, vmax=1)
		ax_c = f.colorbar(contour)

		ax_c.set_label("$P(Species = %0.1i)$"%labels[1])
		ax_c.set_ticks([0, .25, .5, .75, 1])

		ax.scatter(data.loc[data.SPECIES != species,'SEPAL_LENGTH'], data.loc[data.SPECIES != species,'PETAL_WIDTH'], 
			c=data.loc[data.SPECIES != species,'SPECIES'], 
			s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
		ax.set(aspect="equal",
		       xlim=(4, 8), ylim=(0, 2.8),
		       xlabel="SEPAL_LENGTH", ylabel="PETAL_WIDTH", title="Logistic Regression Classification Between Species %0.1i and %0.1i" % (labels[0], labels[1]))
		plt.show()

def predict_probability_point(X, Y, point):
	data = pd.concat((X,Y),axis=1)
	a, b = point
	clf = LogisticRegression(random_state=0).fit(data.loc[data.SPECIES != 2,['SEPAL_LENGTH','PETAL_WIDTH']], 
			data.loc[data.SPECIES != 2,'SPECIES'])
	probs = clf.predict_proba(np.array([[a,b]]))[:, 1]
	print("The probability of point (%0.1f,%0.1f) belonging to Species 1 is %0.2f" % (a, b, probs))

def gradient_descent_classification_plot(X, Y, iter):
	data = pd.concat((X,Y),axis=1)
	xx, yy = np.mgrid[4:8:.01, 0:2.8:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	species = 2
	labels = data.loc[data.SPECIES != species,'SPECIES'].unique()
	clf = LogisticRegression(max_iter=1, warm_start = True, solver = 'sag',random_state=0).fit(data.loc[data.SPECIES != species,['SEPAL_LENGTH','PETAL_WIDTH']], 
			data.loc[data.SPECIES != species,'SPECIES'])
	probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))

	for i in range(iter):
		display.clear_output(wait=True)
		time.sleep(0.1)
		contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
		vmin=0, vmax=1)

		ax.scatter(data.loc[data.SPECIES != species,'SEPAL_LENGTH'], data.loc[data.SPECIES != species,'PETAL_WIDTH'], 
		c=data.loc[data.SPECIES != species,'SPECIES'], 
		s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
		ax.set(aspect="equal",
		       xlim=(4, 8), ylim=(0, 2.8),
		       xlabel="SEPAL_LENGTH", ylabel="PETAL_WIDTH", title="Logistic Regression Classification Between Species %0.1i and %0.1i \n (iteration %0.1i)" % (labels[0], labels[1], i+1))
		if(i==0):
			ax_c = f.colorbar(contour)
			ax_c.set_label("$P(Species = %0.1i)$"%labels[1])
			ax_c.set_ticks([0, .25, .5, .75, 1])
		display.display(plt.gcf());
		clf.fit(data.loc[data.SPECIES != species,['SEPAL_LENGTH','PETAL_WIDTH']], 
		data.loc[data.SPECIES != species,'SPECIES'])
		probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
	plt.clf();

def plot_cross_entropy():
	x = np.linspace(0,1,1000)
	cost_one = []
	cost_zero = []
	for i in x:
		cost_one.append(logloss(1,i))
		cost_zero.append(logloss(0,i))
	f, axs = plt.subplots(nrows=1,ncols=2)
	ax = axs[0]
	ax.set(xlabel="$\hat{p}$", ylabel="Cross-Entropy", title="Cross-Entropy Loss for y = 0")
	ax.plot(x,cost_zero)
	ax = axs[1]
	ax.set(xlabel="$\hat{p}$", ylabel="Cross-Entropy", title="Cross-Entropy Loss for y = 1")
	ax.plot(x,cost_one)
	plt.tight_layout()
	plt.show()

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)

def OvR_plot(X, Y):
	data = pd.concat((X,Y),axis=1)
	xx, yy = np.mgrid[4:8:.01, 0:2.8:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	fc = np.zeros((3,xx.shape[0], xx.shape[1]))
	labels = [0,1]
	for species in [0,1,2]:
		new_data = data.copy()
		new_data.loc[new_data.SPECIES == species, "SPECIES"] = 3
		new_data.loc[~np.isin(new_data.SPECIES,3), "SPECIES"] = 0
		clf = LogisticRegression(random_state=0).fit(new_data[['SEPAL_LENGTH','PETAL_WIDTH']], 
			new_data['SPECIES'])
		probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
		f, ax = plt.subplots(figsize=(8, 6))
		contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
		                      vmin=0, vmax=1)
		ax_c = f.colorbar(contour)

		ax_c.set_label("$P(Species = %0.1i)$"%species)
		ax_c.set_ticks([0, .25, .5, .75, 1])

		ax.scatter(new_data['SEPAL_LENGTH'], new_data['PETAL_WIDTH'], 
			c=new_data['SPECIES'], 
			s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
		ax.set(aspect="equal",
		       xlim=(4, 8), ylim=(0, 2.8),
		       xlabel="SEPAL_LENGTH", ylabel="PETAL_WIDTH", title="Logistic Regression Classification Between Species %0.1i and the remaining ones" % (species))
		fc[species] = probs
		plt.show()
	probs = np.argmax(fc,axis=0)
	f, ax = plt.subplots(figsize=(8, 6))
	ax_c.set_ticks([0, .25, .5, .75, 1])
	
		
	ax.set(aspect="equal",
		       xlim=(4, 8), ylim=(0, 2.8),
		       xlabel="SEPAL_LENGTH", ylabel="PETAL_WIDTH", title="Logistic Regression Classification with OvR")
	contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
		                      vmin=0, vmax=1)
	ax.scatter(data['SEPAL_LENGTH'], data['PETAL_WIDTH'], 
			c=data['SPECIES'], 
			s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
	plt.show()

def multinomial_plot(X, Y):
	data = pd.concat((X,Y),axis=1)
	xx, yy = np.mgrid[4:8:.01, 0:2.8:.01]
	grid = np.c_[xx.ravel(), yy.ravel()]
	clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs').fit(data[['SEPAL_LENGTH','PETAL_WIDTH']], 
			data['SPECIES'])
	probs = clf.predict(grid).reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
	                      vmin=0, vmax=1)
	ax.scatter(data['SEPAL_LENGTH'], data['PETAL_WIDTH'], 
		c=data['SPECIES'], 
		s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
	ax.set(aspect="equal",
	       xlim=(4, 8), ylim=(0, 2.8),
	       xlabel="SEPAL_LENGTH", ylabel="PETAL_WIDTH", title="Multinomial Logistic Regression Classification")
	plt.show()

def OvR_vs_multinomial_plot():
	# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
	# License: BSD 3 clause
	# make 3-class dataset for classification
	centers = [[-5, 0], [0, 1.5], [5, -1]]
	X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
	transformation = [[0.4, 0.2], [-0.4, 1.2]]
	X = np.dot(X, transformation)

	for multi_class in ('multinomial', 'ovr'):
	    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
	                             multi_class=multi_class).fit(X, y)

	    # print the training scores
	    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

	    # create a mesh to plot in
	    h = .02  # step size in the mesh
	    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                         np.arange(y_min, y_max, h))

	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, x_max]x[y_min, y_max].
	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	    # Put the result into a color plot
	    Z = Z.reshape(xx.shape)
	    plt.figure()
	    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
	    plt.axis('tight')

	    # Plot also the training points
	    colors = "bry"
	    for i, color in zip(clf.classes_, colors):
	        idx = np.where(y == i)
	        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
	                    edgecolor='black', s=20)

	    # Plot the three one-against-all classifiers
	    xmin, xmax = plt.xlim()
	    ymin, ymax = plt.ylim()
	    coef = clf.coef_
	    intercept = clf.intercept_

	    def plot_hyperplane(c, color):
	        def line(x0):
	            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
	        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
	                 ls="--", color=color)

	    for i, color in zip(clf.classes_, colors):
	        plot_hyperplane(i, color)

	plt.show()
