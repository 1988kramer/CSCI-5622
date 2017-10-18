import numpy as np
import sys
import math

kINSP = np.array([(1, 8, +1),
			   (7, 2, -1),
			   (6, -1, -1),
			   (-5, 0, +1),
			   (-5, 1, -1),
			   (-5, 2, +1),
			   (6, 3, +1),
			   (6, 1, -1),
			   (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
			  (0, 4, +1),     # 1 - B
			  (2, 1, +1),     # 2 - C
			  (-2, -3, -1),   # 3 - D
			  (0, -1, -1),    # 4 - E
			  (2, -3, -1),    # 5 - F
			  ])


def weight_vector(x, y, alpha):
	"""
	Given a vector of alphas, compute the primal weight vector w.
	The vector w should be returned as an Numpy array.
	"""
	
	w = np.zeros(len(x[0]))
	'''
	b = 0
	C = math.pi
	passes = 0
	max_passes = 3
	tol = .001
	# iterate over examples i = 0 to m
	# repeat until KKT conditions are met
	#	choose j randomly from the m-1 other options
	#	update alpha i and j
	# Find w, b based on stationarity conditions
	while passes < max_passes:
		num_changed = 0
		for i in range(0, len(x)):
			Ei = calc_error(alpha, x, y, b, i)
			# what's tol?
			if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
				
				j =  np.random.randint(0,len(x))
				while j == i:
					j = np.random.randint(0, len(x))

				Ej = calc_error(alpha, x, y, b, j)

				ai_old = alpha[i]
				aj_old = alpha[j]

				L, H = findLH(i, j, ai_old, aj_old, y, C)
				if L == H:
					continue

				eta = (2 * np.dot(x[i], x[j])) - np.dot(x[i], x[i]) - np.dot(x[j], x[j])
				if eta >= 0:
					continue

				aj_new = aj_old - y[j] * (((Ei - Ej)) / eta)
				aj_new = constrain_aj(aj_new, L, H)
				if abs(aj_new - aj_old) < 1e-5:
					continue

				ai_new = ai_old + ((y[i] * y[j]) * (aj_old - aj_new))

				alpha[i] = ai_new
				alpha[j] = aj_new
				
				b1 = (b - Ei - y[i]*(alpha[i] - ai_old)*np.dot(x[i], np.transpose(x[i]))
					- y[j]*(alpha[j] - aj_old)*np.dot(x[i], np.transpose(x[j])))
				b2 = (b - Ej - y[i]*(alpha[i] - ai_old)*np.dot(x[i], np.transpose(x[j]))
					- y[j]*(alpha[j] - aj_old)*np.dot(x[j], np.transpose(x[j])))

				if alpha[i] > 0 and alpha[i] < C:
					b = b1
				elif alpha[j] > 0  and alpha[j] < C:
					b = b2
				else:
					b = (b1 + b2) / 2

				num_changed += 1
		if num_changed == 0:
			passes += 1
		else:
			passes = 0
	'''
	for k in range(0, len(x)):
		w += alpha[k] * y[k] * x[k]

	return w

def calc_error(alpha, x, y, b, i):
	f = 0
	for j in range(0, len(x)):
		f += alpha[j] * y[j] * np.dot(x[j], np.transpose(x[i])) + b
	return f - y[i]

def findLH(i, j, ai, aj, y, C):
	L = 0
	H = 0
	if y[i] == y[j]:
		L = max(0, aj - ai)
		H = min(C, C + aj - ai)
	else:
		L = max(0, ai + aj - C)
		H = min(C, aj + ai)
	return L, H

def constrain_aj(aj, L, H):
	if aj < L:
		aj = L
	elif aj > H:
		aj = H
	return aj

def KKT(x, y, alpha, w, b, zeta, i):

	temp = y[i]*(np.add(np.dot(x[i], np.transpose(w)), b))
	if temp < 1 - zeta[i]:
		return False
	if np.sum(alpha * y) != 0:
		return False
	if alpha[i] * (temp - 1 + zeta[i]) != 0:
		return False
	return True

def find_support(x, y, w, b, tolerance=0.001):
	"""
	Given a set of training examples and primal weights, return the indices
	of all of the support vectors as a set.
	"""

	support = set()
	normW = np.linalg.norm(w)
	distances = np.zeros(len(x))
	i = 0
	minDist = sys.float_info.max
	# calculate distance between each point and plane defined by w
	# keeping track of the minimum distance
	# should points on the wrong side be counted?
	for xi, yi in zip(x,y):
		temp = np.add(np.dot(xi, np.transpose(w)), b)
		distances[i] = abs(temp) / normW
		if np.dot(yi, temp) >= 0: # might want to change this to > instead of >=
			if distances[i] < minDist:
				minDist = distances[i]
		# mark instances on the wrong side of the plane as negative
		else:
			distances[i] = distances[i] * -1
		i += 1
	# loop through calculated distances and add indices of all distances that
	# equal minDist to the set of support vectors
	for j in range(0, len(distances)):
		if np.isclose(distances[j], minDist, atol=tolerance):
			support.add(j)

	return support



def find_slack(x, y, w, b):
	"""
	Given a set of training examples and primal weights, return the indices
	of all examples with nonzero slack as a set.
	"""

	slack = set()
	i = 0
	for xi, yi in zip(x,y):
		if yi * np.add(np.dot(xi, np.transpose(w)), b) < 0:
			slack.add(i)
		i += 1
	return slack
