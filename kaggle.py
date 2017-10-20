#Elizabeth E. Esterly & Danny Byrd
#kaggle.py
#last updated 10/19/2017
from naiveBayesImplementation import *

#####writePredictions#########
# outPredictionsFile ::: string, file to write to
# predictions ::: list of lists [id, classPrediction]
# writes a Kaggle-friendly .csv
def writePredictions(predictions, outPredictionsFile):
	with open(outPredictionsFile, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(predictions)

#MAKE A KAGGLE SUBMISSION
############################################
trainFile = 'training.csv'
testFile = 'testing.csv'
training, classCounts = makeTrainingMatrix(trainFile, 0.00001) #trainFile = training data
testing = makeTestingMatrix(testFile)  #testFile = testing data
predictions = classify(testing, training, classCounts) #predictions is a list of lists
writePredictions(predictions, outPredictionsFile = 'predictions.csv')