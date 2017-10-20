#Elizabeth E. Esterly & Danny Byrd
#naiveBayesImplementation.py
#last updated 10/10/2017

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
	df = betaAdjustment(df, beta)  # add hallucinated counts (beta from Dirichlet distribution)
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
		training =  training.mul(nonZero, axis=1)#multiply each row of the training matrix (axis=1) by the current row of test data
		training = training.prod(axis = 1) #and we sum!
		result = training * classCounts #multiply resulting vector by vector of P(Y) (probability that class occurred)
		predictions.append([df.index[i], result.idxmax()]) #append id number, idxmax() gives the row index of the max value from result; the row index corresponds to the class => prediction
	return predictions

#####writePredictions#########
# outPredictionsFile ::: string, file to write to
# predictions ::: list of lists [id, classPrediction]
# writes a Kaggle-friendly .csv
def writePredictions(predictions, outPredictionsFile):
	with open(outPredictionsFile, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(predictions)

#####partitionData##########
# infile ::: data to be partitioned
# outfile1 ::: resulting partition to be used as test
# outfile2 :::  remaining data
# rows::: number of rows in outfile1
def partitionTrainingData(infile, outfile1, outfile2, rows):
	with open(infile, 'rb') as f, open(outfile1, 'wb') as out1, open(outfile2, 'wb') as out2:
		reader = csv.reader(f)
		csv.writer(out1).writerows(itertools.islice(reader, 0, rows))
		csv.writer(out2).writerows(reader)

#####reshapeTrainAsTest
# filetoreshape ::: path to training-style .csv, strips the class data
# expects non-altered data
# returns test ::: dataframe in test matrix format
#         groundTruth ::: classifications for each row of dataframe
def reshapeTrainAsTest(fileToReshape):
	test = makeTestingMatrix(fileToReshape) #make the testing matrix dataframe
	groundTruth = test[61189].tolist() #save the class column into a list
	test = test.drop(61189, axis=1) #remove class column from the dataframe
	return test, groundTruth

#####createConfusionMatrix
# predictions ::: list of [id, class predictions] made by classify()
# groundTruth ::: list of ground truth values made by reshapeTrainAsTest()
# returns df ::: a confusion matrix Dataframe
def createConfusionMatrix(predictions, groundTruth):
	legend = makeLabelDict()
	df = pd.DataFrame(0, index=legend.keys(), columns= legend.keys()) #creates a dataframe of size labels X labels
	p = [element[1] for element in predictions] #grab the classes out of the predictions list and ditch the id numbers
	p.pop(0) #get rid of the 'id, class' that's stored in index 0
	for i in xrange(0, len(p)):
		df.ix[groundTruth[i], p[i]] += 1 #increment the value here, a correct prediction will be on the identity
	return df

####betaAccuracyEvaluation##########
# betaVal ::: adjust weight of prior and evaluate performance
def betaAccuracyEvaluation(betaVal):
	training, classCounts = makeTrainingMatrix(trainFile, betaVal)  # trainFile = training data
	return training, classCounts

#FOR INTERNAL TESTING (CONVERT TRAIN TO TEST)
############################################
'''''
infile = 'training.csv' #file to process
outfile1 = 'training10Test.csv' #file you write to, test (partitioned data)
outfile2 = 'training90.csv' #file you write to, train (partitioned data)
partitionTrainingData(infile, outfile1, outfile2, rows = 1200)
testing, groundTruth = reshapeTrainAsTest(outfile1)
trainFile = outfile2
'''''

#w/internal testing: TEST A RANGE OF BETA VALUES
############################################
'''''
from visualizations import plotAccuracies
betas = np.logspace(.00001, 1, 5)
accuracies = []
for b in betas:
	training, classCounts = betaAccuracyEvaluation(b)
	predictions = classify(testing, training, classCounts)
	p = [element[1] for element in predictions]
	p.pop(0)
	accuracy = 0.0
	for i in xrange(0, len(p)):
		if groundTruth[i] == p[i]:
			accuracy += 1.0
	accuracy /= len(p)
	accuracies.append(accuracy)
plotAccuracies(betas, accuracies)
'''''


#w/internal testing: CREATE CONFUSION MATRIX IMG AND PRINT TO .TXT FILE
############################################
#from visualizations import printConfusionMatrix, heatmap
#training, classCounts = makeTrainingMatrix(trainFile, 1.0) #trainFile = training data
#predictions = classify(testing, training, classCounts) #predictions is a list of lists
#printConfusionMatrix(df = createConfusionMatrix(predictions, groundTruth))
#heatmap(df, dict = makeLabelDict())

#KAGGLE
############################################
trainFile = 'training.csv'
testFile = 'testing.csv'
training, classCounts = makeTrainingMatrix(trainFile, 0.00001) #trainFile = training data
testing = makeTestingMatrix(testFile)  #testFile = testing data
predictions = classify(testing, training, classCounts) #predictions is a list of lists
writePredictions(predictions, outPredictionsFile = 'predictions.csv')







