#Elizabeth E. Esterly & Danny Byrd
#theTester.py
#last updated 10/09/2017
import numpy as np
import csv
from bayesForDays import *

CLASSES = 20
VOCABULARY = 61188

########updateForBeta#######
# takes a beta value and updates the training matrix and total word counts based on it
def updateForBeta(beta, trainingMatrix, classWordCount):
	betaMatrix = np.ones((CLASSES + 1, VOCABULARY + 1))
	np.add(betaMatrix, trainingMatrix) #add beta to the 0 counts
	divisor = []
	divisor.append(0)
	for i in range (1, CLASSES + 1):
		classWordCount[i] += beta #add beta to the total word counts for each class
		for j in range(1, VOCABULARY + 1):
			trainingMatrix[i, j] = float(trainingMatrix[i,j])/classWordCount[i]
	return trainingMatrix, classWordCount

#########writePredictions#########
# predictions: a list of predictions: id, class
def writePredictions(predictions):
	with open('predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow('id,class')
		for p in predictions:
			writer.writerow(p)
	return

##########makePredictionMatrix#####
# read test data into a matrix.
def predict(trainingMatrix, classToPrior):
	predictions = []
	with open('testing.csv', 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			max = -1
			idx = -1
			dataToLabel = np.ones(1, VOCABULARY + 1)
			for i in range(1, VOCABULARY + 1):
				dataToLabel[0, i] += int(row[i])
			for x in range(1, CLASSES + 1):
				val = maxVal(dataToLabel, trainingMatrix[x,])
				if val > max:
					max = val
					idx = x
		predictions.append((row, idx))
	return predictions

def maxVal(pred, trained, prior):
	val = prior * np.multiply(pred, trained)
	return val


