#Elizabeth E. Esterly & Danny Byrd
#theTester.py
#last updated 10/08/2017
import numpy as np
import csv
from bayesForDays import classReference

CLASSES = 20
VOCABULARY = 61188
###########################################
BETA = 1/VOCABULARY #for testing part 1 only!!!!!! "Hallucinating" one count of a word

predictions = []
with open ('testing.csv') as f:
	reader = csv.reader(f)
	for row in reader: #for each data point to predict
		#make an array and initialize with log likelihoods for all classes, then just keep summing as we see more entries
		classComparisons = np.zeros((1, CLASSES + 1))
		#TODO: P(Y) goes here as the first sum, grab from classReference
		for i in range(1, VOCABULARY + 1):
			if row[i] != '0':
				classComparisons = (c + np.log(BETA) for c in classComparisons) #add the log if we hallucinated a value
			else:
				#TODO: calculate the log likelihood for each entry and sum--look at adjusting denominator for "hallucinated" values


		#write res on the fly?

#read in line: for each line:
#for each class:
#for i in range (1, l)
#log(class prior) +