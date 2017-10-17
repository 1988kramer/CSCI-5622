import numpy as np
import sys

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
    # iterate over examples i = 0 to m
    # repeat until KKT conditions are met
    #	choose j randomly from the m-1 other options
    #	update alpha i and j
    # Find w, b based on stationarity conditions
    
    return w

def calcAlphas(ai_old, aj_old, xi, xj, yi, yj, w, b):
	# not super confident about the error calculation here
	Ei = np.add(np.dot(xi, np.transpose(w)), b) - yi
	Ej = np.add(np.dot(xj, np.transpose(w)), b) - yj

	eta = (2 * xi * xj) - (xi * xi) - (xj * xj)

	aj_new = aj_old - yi * (((Ei - Ej)) / eta)
	ai_new = ai_old + ((yi * yj) * (aj_old - aj_new))

	return ai_new, aj_new

def KKT(x, y, alpha, w, b, zeta, i):
	temp = y[i]*(np.add(np.dot(x[i], np.transpose(w)), b))
	if temp < 1 - zeta[i]:
		return False
	if np.sum(alpha * y) != 0:
		return False
	if alpha[i] * (temp + zeta[i]) != 0:
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
