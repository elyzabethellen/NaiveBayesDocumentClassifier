#Elizabeth E. Esterly & Danny Byrd
#bayesForDays.py
#last updated 10/09/2017

import numpy as np
import csv

CLASSES = 20
VOCABULARY = 61188

########makeTrainingMatrix##############
# returns
# 1) classCount : array of 21 positions holding a dummy count at classCount[0] and counts
# for how many times class is seen in each corresponding array position
# 2) trainingDataCount : total count of training data points
# 3) trainingDataMatrix:
# results in matrix of 21 rows and 61189 columns, with row and col 0 being 0'd out
#####################################################
# index 0 should be a dummy value at beginning of each new list
#  as when test data is run it should line up, i.e val for feature 1 is in index 1
#                feat  1 2 3 4 5
#  no 0th label   #  0 0 0 0 0 0
#  1st label      #  0 2 8 9 0 1
#  2nd label      #  0 1 2 1 8 2
#  3rd label      #  0 4 0 2 6  . . .
def makeTrainingMatrix():
	trainingMatrix = np.zeros((CLASSES + 1, VOCABULARY + 1))
	classCount = np.ones((1, CLASSES + 1))
	with open('training.csv') as t:
		reader = csv.reader(t)
		for row in reader:
			idx = row[-1] #last position: get what class it belongs to
			classCount[0, idx] += 1 #keep track here-sum() will give us the wrong val because of class at end
			for i in range(1, VOCABULARY + 1):
				trainingMatrix[idx, i] += int(row[i]) #increment the class at that vocab column with the count
	trainingDataCount = np.sum(classCount)
	return classCount, trainingDataCount, trainingMatrix

########makeDicts##############
# takes classCount, trainingDataCount, trainingMatrix (from makeTrainingMatrix)
# returns 3 dicts:
# classToLabel, classToTotalWordCount, classToPrior
def makeDicts(classCount, trainingDataCount, trainingMatrix):
	classToLabel = {}
	classToTotalWordCount = {}
	classToPrior = {}

	#dictionary |  class number : label
	with open('newsgrouplabels.txt')as f:
		for line in f:
			y = int(''.join(x for x in line if x.isdigit()))
			z = ''.join(x for x in line if not x.isdigit() and not x == '\n')
			classToLabel[y] = z

	for i in range(1, CLASSES +1):

		# dictionary |  class number : total number of words in this classification (use in denominator)
		classToTotalWordCount[i] = np.sum(trainingMatrix[i,])

		# dictionary |  class number : probability of this class
		classToPrior[i] = float(classCount[0, i])/trainingDataCount

	return classToLabel, classToTotalWordCount, classToPrior

'''''
print classToLabel
print classToTotalWordCount
print classToPrior
'''''