#Elizabeth E. Esterly & Danny Byrd
#transformation.py
#last updated 10/19/2017

import itertools
from naiveBayesImplementation import *

#####partitionTrainingData##########
# infile ::: data to be partitioned
# outfile1 ::: resulting partition to be used as test
# outfile2 :::  remaining data
# rows::: number of rows in outfile1
def partitionTrainingData(infile, outfile1, outfile2, rows):
	with open(infile, 'rb') as f, open(outfile1, 'wb') as out1, open(outfile2, 'wb') as out2:
		reader = csv.reader(f)
		csv.writer(out1).writerows(itertools.islice(reader, 0, rows))
		csv.writer(out2).writerows(reader)
		return

#####reshapeTrainAsTest
# filetoreshape ::: path to training-style .csv, strips the class data
# expects non-altered data
# returns test ::: dataframe in test matrix format
# groundTruth ::: classifications for each row of dataframe
def reshapeTrainAsTest(fileToReshape):
	test = makeTestingMatrix(fileToReshape)  # make the testing matrix dataframe
	groundTruth = test[61189].tolist()  # save the class column into a list
	test = test.drop(61189, axis=1)  # remove class column from the dataframe
	return test, groundTruth


