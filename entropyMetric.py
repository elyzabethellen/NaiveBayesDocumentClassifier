
totalWords = 15

def buildIdList(TM):

	#print training.index
	data = TM.sum() # this adds those probabilities  # this is currently the metric 
	

	dataList = [x for _, x in sorted(zip(data.values,data.index), key=lambda pair: pair[0], reverse= True)]
	dataList = dataList[:totalWords]
	vocab = vocabData()
	print len(vocab)
	print map((lambda x: vocab[(x-2)].rstrip()), dataList)
	#print dataList

##### builds dictionary of words  #####
def vocabData():
	words = []
	with open('vocabulary.txt')as f:
		for line in f:			
			words.append(line)
	return words
