#Elizabeth E. Esterly & Danny Byrd
#bayesForDays.py
#last updated 10/08/2017

import numpy as np
import csv
import time
CLASSES = 20
VOCABULARY = 61188

#create a dictionary of int(class number) to list of attributes by reading in text file (for printing results)
classReference = {}
with open('newsgrouplabels.txt')as f:
	for line in f:
		y = int(''.join(x for x in line if x.isdigit()))
		z = ''.join(x for x in line if not x.isdigit() and not x == '\n')
		label = []
		label.append(z)
		classReference[y] = label
'''
print "Dictionary of labels created:"
print classReference
'''
'''
print "\n Begin training at"
print time.ctime()
'''

########DATA PROCESSING##############
# results in matrix of 21 rows and 61189 columns, with row and col 0 being 0'd out
#####################################################

# index 0 should be a dummy value at beginning of each new list
#  as when test data is run it should line up, i.e val for feature 1 is in index 1
#                feat  1 2 3 4 5
#  no 0th label   #  0 0 0 0 0 0
#  1st label      #  0 2 8 9 0 1
#  2nd label      #  0 1 2 1 8 2
#  3rd label      #  0 4 0 2 6  . . .
trainingMatrix = np.zeros((CLASSES + 1, VOCABULARY + 1))
seen = np.zeros((1, CLASSES + 1))
with open('training.csv') as t:
	reader = csv.reader(t)
	for row in reader:
		idx = row[-1] #last position: get what class it belongs to
		seen[0, idx] += 1
		for i in range(1, VOCABULARY + 1):
			if row[i] != '0': #data comes in as a string
				trainingMatrix[idx, i] += int(row[i]) #increment the class at that vocab column with the count
trainingDataCount = np.sum(seen)
print trainingDataCount

'''''
print "Training matrix completed at"
print time.ctime()
print "Training matrix sample--first row of matrix--should be all zeroes \n"
print trainingMatrix[0,]
print "Training matrix sample--index row 10 of matrix"
print trainingMatrix[10,]
'''''
#update dictionary with
# class number : ('newsgroup.names', totalWordCount (this class), P(this class) = seen[label]/ trainingDataCount)
for i in range(1, CLASSES +1):
	classReference.get(i).append(np.sum(trainingMatrix[i,]))
	classReference.get(i).append(float(seen[0, i])/trainingDataCount)
print classReference


