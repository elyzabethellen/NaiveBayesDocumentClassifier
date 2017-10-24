#Elizabeth E. Esterly & Danny Byrd
#entropyMetric.py
#last updated 10/23/2017
import numpy as np

totalWords = 100

########boostMatrixData##############
# given 2 lists, and a training matrix this weights words up or down depending on what word they are 
# matrix : a computed training matrix which we can use to calculate conditional entropy
# boostList: a list of words to be increased in the matrix
# thinList: a list of words to be reduced in this matrix 
def boostMatrixData(trainingMatrix,boostList,thinList):
	trainingMatrix[boostList] = trainingMatrix[boostList] * 4
	trainingMatrix[thinList] = trainingMatrix[thinList] * 0.25
	return trainingMatrix

########calculateConditionalEntropy##############
# calculates the conditional entropy for a training matrix 
# matrix : a computed training matrix which we can use to calculate conditional entropy
# CC: class probabilities
def calculateConditionalEntropy(matrix,CC):
	entropyValuesCalculated = ((matrix+0.0000001) * -(matrix+ 0.0000001).apply(np.log)) * CC
	# changes each cell in the matrix to match the conditional entropy 

	sumData = entropyValuesCalculated.sum()
	# sum these values together by column (word)

	# build the list of ids
	reducedList,lowestEntropyValues = buildIdList(entropyValuesCalculated.sum())	
	return reducedList,lowestEntropyValues
	


########buildIdList##############
# gets the list of id values sorted by lowest entropy and capped at totalWords variable
# data : takes in list of ids with calculated entropy
def buildIdList(data):
	dataList = [x for _, x in sorted(zip(data.values,data.index), key=lambda pair: pair[0], reverse= False)]
	reducedList = dataList[:totalWords]
	flippedList = buildFlippedIdList(data)
	return reducedList,flippedList

########buildFlippedIdList##############
# gets the highest entropy words
# data : takes in list of ids with calculated entropy
def buildFlippedIdList(data):
	dataList = [x for _, x in sorted(zip(data.values,data.index), key=lambda pair: pair[0], reverse= True)]
	reducedList = dataList[:totalWords]
	return reducedList

########buildIdList##############
# takes a list of ids, and maps vocabulary to them
# dataList : list of vocab ids
def printVocabWords(dataList):
	vocab = vocabData()
	print map((lambda x: vocab[(x-1)].rstrip()), dataList)

########vocabData##############
# loads data from the vocabulary for display
def vocabData():
	words = []
	with open('vocabulary.txt')as f:
		for line in f:			
			words.append(line)
	return words
