import os
import matplotlib.pyplot as plt
import numpy as np
#mpl.rc('axes', labelsize=14)
#mpl.rc('xtick', labelsize=12)
#mpl.rc('ytick', labelsize=12)

def save(fig_id,fig_extension="png",resolution=300,directorio_root=".",tight_layout=False):
	directorio_root
	imagenes_path = os.path.join(directorio_root, "imagenes")
	path = os.path.join(imagenes_path, fig_id + "." + fig_extension)
	if not os.path.isdir(imagenes_path):
		os.makedirs(imagenes_path)
	print("Saving figure", fig_id)
	if tight_layout:
		plt.tight_layout()
	n=0
	while(os.path.isfile(path)):
		n=n+1
		path=os.path.join(imagenes_path, fig_id +str(n)+ "." + fig_extension)
		
	plt.savefig(path, format=fig_extension, dpi=resolution)

def visual(models,titles,x_test,y_labels):
	def make_meshgrid(x, y, h=.02):
		"""Create a mesh of points to plot in

		parameters
		
		x: data to base x-axis meshgrid on
		y: data to base y-axis meshgrid on
		h: stepsize for meshgrid, optional

		Returns
		-------
		xx, yy : ndarray
		"""
		x_min, x_max = x.min() - 1, x.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		np.arange(y_min, y_max, h))
		return xx, yy


	def plot_contours(ax, clf, xx, yy, **params):
		"""Plot the decision boundaries for a classifier.

		Parameters
		----------
		ax: matplotlib axes object
		clf: a classifier
		xx: meshgrid ndarray
		yy: meshgrid ndarray
		params: dictionary of params to pass to contourf, optional
		"""
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)
		return out


	# Set-up 2x2 grid for plotting.
	fig, sub = plt.subplots(2, 2)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	X0, X1 = x_test[:, 0], x_test[:, 1]
	xx, yy = make_meshgrid(X0, X1)

	for clf, title, ax in zip(models, titles, sub.flatten()):
		plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.4)
		ax.scatter(X0, X1, alpha=0.5, c=y_labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xlabel('Component x')
		ax.set_ylabel('Component y')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title, fontsize=8)
	save("clasificador")
	plt.show(block=False)



import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
