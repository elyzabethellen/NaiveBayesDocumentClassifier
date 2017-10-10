#Elizabeth E. Esterly & Danny Byrd
#naiveBayesImplementation.py
#last updated 10/10/2017

import pandas as pd

########makeTrainingMatrix##############
# read the .csv data to a pandas dataframe
# row indices = class label
# sum across class labels and consolidate data
# return training, classCounts
def makeTrainingMatrix(beta):
	df = pd.read_csv('training.csv', index_col=61189)  # 11 min; 11 min again; 12 min
	classCounts = df.index.value_counts()
	df = df.groupby(df.index).sum()
	df = betaAdjustment(df, beta)
	df = (df.T / df.T.sum()).T
	training = df.drop('1', 1)
	return training, classCounts

########betaAdjustment###############
# fill in the beta val to use as dirichlet prior
def betaAdjustment(training, beta):
	return training + beta

########makeTrainingMatrix##############
# read the .csv data to a pandas dataframe
# row indices = id label
# return testing
def makeTestingMatrix(beta):
	testing = pd.read_csv('testing.csv', index_col=1)
	testing = betaAdjustment(testing, beta)
	return testing

########makeLabelDict##############
# classNumber mapped to NewsgroupNameLabel
def makeLabelDict():
	classToLabel = {}
	#dictionary |  class number : label
	with open('newsgrouplabels.txt')as f:
		for line in f:
			y = int(''.join(x for x in line if x.isdigit()))
			z = ''.join(x for x in line if not x.isdigit() and not x == '\n')
			classToLabel[y] = z
	return classToLabel

#######classify#############
def classify(df, training):
	predictions = []
	for i in xrange(0, len(df.index)): #for each data point
		result =  training.mul(df.iloc[i]).sum() #multiply df row by training data and sum rows (dot product)
		predictions.append(result.idxmax())
	return predictions



training, classCounts = makeTrainingMatrix(1)
testing = makeTestingMatrix(1)
predictions = classify(testing, training)
print predictions






