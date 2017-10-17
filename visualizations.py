#Elizabeth E. Esterly & Danny Byrd
#visualizations.py
#last updated 10/04/2017
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
def plotBetaAccuracies():
	betas = np.linspace()
	accuracies = []
	for b in betas:
		accuracies.append(betaAccuracyEvaluation(b))
	plt.plot(betas, accuracies)
	plt.savefig("leTest.jpg")
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
	fig.savefig('confusion.jpg', dpi=300)
	return