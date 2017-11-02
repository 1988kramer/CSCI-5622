import argparse
import numpy as np

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class FoursAndNines:
	"""
	Class to store MNIST data
	"""

	def __init__(self, location):
		# You shouldn't have to modify this class, but you can if
		# you'd like.

		import pickle, gzip

		# Load the dataset
		f = gzip.open(location, 'rb')

		train_set, valid_set, test_set = pickle.load(f)

		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]

		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]

		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]

		f.close()

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname:
		plt.savefig(outname)
	else:
		plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
						help="Restrict training to this many examples")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

	# TODO: Use the Sklearn implementation of support vector machines to train a classifier to
	# distinguish 4's from 9's (using the MNIST data from the KNN homework).
	# Use scikit-learn's Grid Search (http://scikit-learn.org/stable/modules/grid_search.html) to help determine
	# optimial hyperparameters for the given model (e.g. C for linear kernel, C and p for polynomial kernel, and C and gamma for RBF).
	
	# need to add polynomial
	'''
	tuned_params = [{'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [2, 3, 4]}]

	scores = ['precision']

	for score in scores:
		print("# tuning hyper-parameters for %s" % score)
		print()
		classifier = GridSearchCV(SVC(), tuned_params, cv = 5,
								  scoring='%s_macro' % score)
		classifier.fit(data.x_train, data.y_train)
		print("Best parameters set found on development set:")
		print()
		print(classifier.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = classifier.cv_results_['mean_test_score']
		stds = classifier.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				% (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_pred = classifier.predict(data.x_test)
		print(classification_report(data.y_test, y_pred))
		print()
	'''
	# -----------------------------------
	# Plotting Examples
	# -----------------------------------
	# use svc.get_params to get weights
	# Display in on screen
	
	lin = SVC(kernel='linear', C=1)
	RBF = SVC(kernel='rbf', gamma=.001, C=1000)
	poly = SVC(kernel='poly', degree=2, C=1000)
	print("training linear")
	lin.fit(data.x_train, data.y_train)
	print("training RBF")
	RBF.fit(data.x_train, data.y_train)
	print("training polynomial")
	poly.fit(data.x_train, data.y_train)
	print("testing")
	lin_pred = lin.predict(data.x_test)
	RBF_pred = RBF.predict(data.x_test)
	poly_pred = poly.predict(data.x_test)
	print()
	print("linear results")
	print(classification_report(data.y_test, lin_pred))
	print()
	print("RBF results")
	print(classification_report(data.y_test, RBF_pred))
	print()
	print("poly results")
	print(classification_report(data.y_test, poly_pred))

	

	# Plot image to file
	#mnist_digit_show(data.x_train[1,:], "mnistfig.png")
