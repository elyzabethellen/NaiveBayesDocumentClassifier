###Testing scratch file to ensure that functions work the way we think they do........
import numpy as np
from visualizations import confusionMatrix

mTest = np.matrix('1, 2 ; 3, 4')
confusionMatrix(mTest)

x = [1, 2, 3, 4]

x = (x + 1 for x in x)
print list(x)