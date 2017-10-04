#Elizabeth E. Esterly & Danny Byrd
#visualizations.py
#last updated 10/03/2017
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####confusionMatrix########
def confusionMatrix(m):
	df = pd.DataFrame(m)
	np.savetxt('confusion.txt', df.values, fmt='%d')
	return

####betaIteration##########
def betaAccuracyEvaluation(betaVal):
	accuracy = None
	return accuracy

###plotBetaAccuracies######
def plotBetaAccuracies(betas):
	accuracies = []
	for b in betas:
		accuracies.append(betaAccuracyEvaluation(b))
	plt.plot(betas, accuracies)
	plt.savefig("leTest.jpg")
	plt.show()

