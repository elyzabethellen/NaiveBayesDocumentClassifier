#Elizabeth E. Esterly & Danny Byrd
#theTester.py
#last updated 10/09/2017
import numpy as np
import csv
from bayesForDays import classReference # class number : ('newsgroup.names', totalWordCount (this class), P(this class) = seen[label]/ trainingDataCount)


CLASSES = 20
VOCABULARY = 61188
###########################################
#TODO:Make this into a function that takes BETA as a value....

BETA = 1/VOCABULARY #for testing part 1 only!!!!!! "Hallucinating" one count of a word

predictions = []
with open ('testing.csv') as f:
	reader = csv.reader(f)
	for row in reader: #for each data point to predict
		#make an array and initialize with log likelihoods for all classes, then just keep summing as we see more entries
		#sum in place in the array, where position is equivalent to class.
		classComparisons = np.zeros((1, CLASSES + 1))
		for i in range(1, CLASSES + 1):
			classComparisons[i] += np.log(classReference.get(i)[2])# log P(Y) goes in all slots as the first sum, grab from classReference
		for i in range(1, VOCABULARY + 1):
			if row[i] != '0':
				classComparisons = (c + np.log(BETA) for c in classComparisons) #add the log of beta to each slot if we hallucinated a value
			else:
				#TODO: calculate the log likelihood for each entry and sum--look at adjusting numerator AND denominator for "hallucinated" values
