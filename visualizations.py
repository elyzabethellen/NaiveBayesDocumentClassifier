#Elizabeth E. Esterly & Danny Byrd
#visualizations.py
#last updated 10/19/2017
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####confusionMatrix########
# takes a dataframe and prints to a text file
# m ::: a 2-d array (matrix)
def printConfusionMatrix(df):
	np.savetxt('confusion.txt', df.values, fmt='%d')
	return

###plotBetaAccuracies######
# betas  :::  a list of beta values to be evaluated
def plotAccuracies(betas, accuracies):
	plt.semilogx(betas, accuracies, 'co', ms= 10) #log axis for x, cyan markers, size 15
	plt.grid(True)
	plt.margins(0.1)
	plt.xlabel('beta')
	plt.ylabel('accuracy')
	plt.savefig('beta.jpg', dpi = 300)
	print 'Accuracy values for beta ='
	print accuracies
	plt.show()

###heatmap################
# df ::: a dataframe to be visualized
# dict ::: a dictionary of numerical class values to their text values
# saves the figure locally
def heatmap(df, dict):
	fig, ax = plt.subplots()
	fig.subplots_adjust(left=0.3)
	im = ax.imshow(df, interpolation='nearest', cmap=plt.cm.ocean)
	ax.set_xticks(range(len(df.columns)))
	ax.set_yticks(range(len(df.index)))
	ax.set_xticklabels(dict.keys())
	ax.set_yticklabels(df.index)
	fig.colorbar(im, ax=ax)
	fig.show()
	#fig.savefig('confusion.jpg', dpi=300)
