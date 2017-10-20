#Elizabeth E. Esterly & Danny Byrd
#naiveBayesImplementation.py
#last updated 10/19/2017

import pandas as pd
import csv
import numpy as np
import itertools

########makeTrainingMatrix##############
# read the .csv data to a pandas dataframe
# row indices = class label
# sum across class labels and consolidate data
# return training, classCounts
def makeTrainingMatrix(filename, beta):
	df = pd.read_csv(filename, header = None, index_col=61189)
	df.index.name = 'class'
	classCounts = df.index.value_counts() #how many times a class was seen, for probablilites
	classCounts = classCounts.T / classCounts.T.sum() #make into probs
	classCounts = classCounts.sort_index() # sort by class index
	df = df.groupby(df.index).sum() #group all classes together and sum their columns by class--row index becomes class
	x = df.columns.tolist() #get a list of all columns
	df = df.drop(x[0], axis =1) #drop the id number column, no calculations on that please!
	#df = betaAdjustment(df, beta)  # add hallucinated counts (beta from Dirichlet distribution)
	return df, classCounts

########betaAdjustment###############
# training ::: training matrix dataframe
# beta ::: value for beta
# add beta val to use as dirichlet prior to each df entry and divide by row-wise sum to get probability
def betaAdjustment(training, beta):
	training = training + beta #plus operator acts elementwise on a dataframe
	return training.div(training.sum(axis=1), axis =0)  #probabilities: divide by each row sum of elements

########makeTrainingMatrix##############
# read the .csv data to a pandas dataframe
# row indices = id label
# return testing :: dataframe from .csv, unaltered
def makeTestingMatrix(filename):
	testing = pd.read_csv(filename, header =None, index_col=0)
	return testing

#######classify#############
# df ::: testing matrix
# t  ::: training matrix
# classCounts ::: list of times class appeared (for P(Y))
# return predictions ::: list of lists [id, classPrediction]
def classify(df, t, classCounts):
	predictions = []
	predictions.append(['id','class'])
	for i in xrange(0, len(df.index)): #for each data point in the test data
		training = t #copy the training matrix
		row = df.iloc[[i]] #get our current test data row
		nonZero = row.loc[:, (row != 0).all()]#get all non-zero columns values from test data row; double [[]] to force df, not series
		cols = nonZero.columns.tolist() #list the columns that have values
		training = training[cols] #slice training copy with columns from nonZero
		training =  training.mul(list(nonZero), axis=1)#multiply each row of the training matrix (axis=1) by the current row of test data
		training = training.prod(axis = 1) #and we sum!
		result = training * classCounts #multiply resulting vector by vector of P(Y) (probability that class occurred)
		predictions.append([df.index[i], result.idxmax()]) #append id number, idxmax() gives the row index of the max value from result; the row index corresponds to the class => prediction
	return predictions










