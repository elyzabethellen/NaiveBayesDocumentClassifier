#Elizabeth E. Esterly & Danny Byrd
#evaluation.py
#last updated 10/19/2017

import numpy as np
import pandas as pd
from time import ctime
from transformation import *
from visualizations import *
from naiveBayesImplementation import *

####betaAccuracyEvaluation##########
# betaVal ::: adjust weight of prior and evaluate performance
def betaAccuracyEvaluation(betaVal):
	training, classCounts = makeTrainingMatrix(trainFile, betaVal)  # trainFile = training data
	return training, classCounts

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

#####createConfusionMatrix
# predictions ::: list of [id, class predictions] made by classify()
# groundTruth ::: list of ground truth values made by reshapeTrainAsTest()
# returns df ::: a confusion matrix Dataframe
def createConfusionMatrix(predictions, groundTruth):
	legend = makeLabelDict()
	df = pd.DataFrame(0, index=legend.keys(), columns= legend.keys()) #creates a dataframe of size labels X labels
	for i in xrange(0, len(p)):
		df.ix[groundTruth[i], p[i]] += 1 #increment the value here, a correct prediction will be on the identity
	return df


#FOR INTERNAL TESTING (CONVERT TRAIN TO TEST)
############################################
print 'Converting training data to 90% train / 10% test.'
print ctime()
infile = 'training.csv' #file to process
outfile1 = '10percenttest.csv' #file you write to, test (partitioned data)
outfile2 = '90percenttrain.csv' #file you write to, train (partitioned data)
partitionTrainingData(infile, outfile1, outfile2, rows = 1200)
print 'Conversion complete.'
print ctime()
print 'Creating testing matrix.'
testing, groundTruth = reshapeTrainAsTest(outfile1)
trainFile = outfile2
print 'Testing matrix complete.'
print ctime()

#TEST A RANGE OF BETA VALUES AND PLOT ACCURACY
############################################
print 'Creating training matrix. This is the longest operation and takes around 10 minutes.'
betas = np.logspace(.00001, 1, 5)
accuracies = []
predictions = []
t, classCounts = makeTrainingMatrix(trainFile, 0) #0 beta adjustment as we will call beta adjustment directly below
print 'Training matrix complete.'
print ctime()
print "Testing 5 logarithmically spaced beta values between 0.0001 and 1."
for b in betas:
	training = t #copy so we can update with new beta without creating new matrix
	training = betaAdjustment(training, b)
	pre = classify(testing, training, classCounts)
	p = [element[1] for element in pre]
	p.pop(0)
	if b == .00001:
		predictions = p
		print 'PARTY TIME'
	accuracy = 0.0
	for i in xrange(0, len(p)):
		if groundTruth[i] == p[i]:
			accuracy += 1.0
	accuracy /= len(p)
	accuracies.append(accuracy)
print "A plot will now appear. Close the plot to continue."
plotAccuracies(betas, accuracies)
print ctime()

#CREATE CONFUSION MATRIX IMG AND PRINT TO .TXT FILE
############################################
print "Creating confusion matrix."
#printConfusionMatrix(df = createConfusionMatrix(predictions, groundTruth))
df = createConfusionMatrix(predictions, groundTruth)
print "A plot will now appear. Close the plot to quit."
heatmap(df, dict = makeLabelDict())
print ctime()
