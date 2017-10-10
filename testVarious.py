###Testing scratch file to ensure that functions work the way we think they do........
import numpy as np
from visualizations import *
from bayesForDays import *
from theTester import *

CLASSES = 20
VOCABULARY = 61188

classCount, trainingDataCount, trainingMatrix = makeTrainingMatrix()
classToLabel, classWordCount, classToPrior = makeDicts(classCount, trainingDataCount, trainingMatrix)

beta = 1
trainingMatrix, classWordCount = updateForBeta(beta, trainingMatrix, classWordCount)

result = predict(trainingMatrix, classToPrior)
print result
