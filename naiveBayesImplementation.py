#Elizabeth E. Esterly & Danny Byrd
#naiveBayesImplementation.py
#last updated 10/10/2017

import pandas as pd
import csv

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
	df = df.groupby(df.index).sum() #group all classes together and sum their columns by class
	df = betaAdjustment(df, beta) #add hallucinated counts (beta from Dirichlet distribution)
	x = df.columns.tolist()
	df = df.drop(x[0], axis =1)
	df = (df.T / df.T.sum()).T #probabilities: divide by each row sum of elements
	return df, classCounts

########betaAdjustment###############
# fill in the beta val to use as dirichlet prior
def betaAdjustment(training, beta):
	return training + beta

########makeTrainingMatrix##############
# read the .csv data to a pandas dataframe
# row indices = id label
# return testing
def makeTestingMatrix(filename, beta):
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
def classify(df, t, classCounts):
	predictions = []
	predictions.append(['id','class'])
	for i in xrange(0, len(df.index)): #for each data point
		training = t
		row = df.iloc[[i]]
		nonZero = row.loc[:, (row != 0).all()]#get all non-zero columns--double [[]] to force df, not series
		cols = nonZero.columns.tolist()
		training = training[cols] #slice training using columns from nonZero
		training = training.multiply(list(nonZero)) #get probs across removed (AXIS = 1)
		training = training.prod(axis=1)
		result = training * classCounts
		predictions.append([df.index[i], result.idxmax()])
	return predictions

#####writePredictions#########
# writing helper function that expects a list of lists
# and writes a Kaggle-friendly .csv
def writePredictions(predictions):
	with open('predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerows(predictions)

#####partitionData##########
# infile: data to be partitioned
# outfile: resulting partition
# maxRows: max rows in outfile
def partitionTrainingData(infile, outfile, maxRows):
	with open(infile) as f:
		with open(outfile, 'w') as c:
			reader = csv.reader(f)
			writer = csv.writer(c)
			for i in range(0, maxRows):
				writer.writerow(reader.next())

#####createConfusionMatrix
# returns a confusion matrix
def createConfusionMatrix():
	legend = makeLabelDict()
	df = pd.DataFrame(0, index=legend.values(), columns= legend.keys())
	return df

####UNCOMMENT TO PARTITION DATA
#infile = 'testing.csv' #file to process
#outfile = 'testing50entries.csv' #file you write to (partitioned data)
#partitionTrainingData(infile, outfile, 50)

#MAIN SCRIPT
############################################
#trainFile = 'trainingFirst10Percent.csv'
#testFile = 'testing10entriesonly.csv'
#outPredictionsFile = 'predictions.csv'
#training, classCounts = makeTrainingMatrix(trainFile, 1.0) #trainFile = training data
#testing = makeTestingMatrix(testFile, 1.0)  #testFile = testing data
#predictions = classify(testing, training, classCounts) #predictions is a list of lists

####UNCOMMENT TO CREATE AND PRINT CONFUSION MATRIX TO .TXT FILE
from visualizations import printConfusionMatrix
df = createConfusionMatrix()
printConfusionMatrix(df)

####UNCOMMENT TO WRITE TO KAGGLE_CSV
#writePredictions(predictions, outPredictionsFile) #write predictions to a kaggle-friendly .csv







