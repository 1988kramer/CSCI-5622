import os
import json
from csv import DictReader, DictWriter

import numpy as np
import re
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score

SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features

# TODO: Add custom feature transformers for the movie review data
class BagOfWordsifier(BaseEstimator, TransformerMixin):

	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		features = [dict() for x in range(len(examples))]
		i = 0
		for ex in examples:
			word_vector = re.compile('\b\w+\w').split(str(ex))
			for word in word_vector:
				if word in features[i].keys():
					features[i][word] += 1
				else:
					features[i][word] = 1
			i += 1
		return features

class NegativeWordCounter(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		text_file = open("negative.txt", "r")
		negativeString = text_file.read()
		negativeWords = negativeString.split("\n")
		features = np.zeros((len(examples), 1))
		i = 0
		for ex in examples:
			exString = str(ex).lower()
			for word in negativeWords:
				features[i, 0] += exString.count(word)
			i += 1
		return features

class VocabularyCounter(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		features = [dict() for x in range(len(examples))]
		i = 0
		for ex in examples:
			word_vector = re.compile('\b\w+\w').split(str(ex))
			for word in word_vector:
				if word in features[i].keys():
					features[i][word] += 1
				else:
					features[i][word] = 1
			i += 1
		vocab = np.zeros((len(examples), 1))
		i = 0
		for feature in features:
			vocab[i] = len(feature.keys())
			i += 1
		return vocab

class BagOfBigramsifier(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		features = [dict() for x in range(len(examples))]
		bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\w', min_df=1)
		analyze = bigram_vectorizer.build_analyzer()
		i = 0
		for ex in examples:
			bigram_vector = analyze(str(ex))
			for bigram in bigram_vector:
				if bigram in features[i].keys():
					features[i][bigram] += 1
				else:
					features[i][bigram] = 1
			i += 1
		return features

class BagOfTrigramsifier(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		features = [dict() for x in range(len(examples))]
		trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), token_pattern=r'\b\w+\w', min_df=1)
		analyze = trigram_vectorizer.build_analyzer()
		i = 0
		for ex in examples:
			trigram_vector = analyze(str(ex))
			for trigram in trigram_vector:
				if trigram in features[i].keys():
					features[i][trigram] += 1
				else:
					features[i][trigram] = 1
			i += 1
		return features

class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('word_bag', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('bag_of_words', BagOfWordsifier()),
            	('vect', DictVectorizer())
            ])),
            ('vocab_count', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('vocab', VocabularyCounter())
            ])),
            ('negative_words', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('negative_count', NegativeWordCounter())
            ])),
            ('bagOfBiGrams', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('bag_of_bigrams', BagOfBigramsifier()),
            	('vect', DictVectorizer())
            ])),
            ('bagOfTriGrams', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('bag_of_trigrams', BagOfTrigramsifier()),
            	('vect', DictVectorizer())
            ]))
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()
    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })


    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier
