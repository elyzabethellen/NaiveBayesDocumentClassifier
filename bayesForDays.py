#Elizabeth E. Esterly & Danny Byrd
#bayesForDays.py
#last updated 10/05/2017

#create a dictionary of class number to group by reading in text file (for printing results)


# create master list with 0s as first list

########DATA PROCESSING##############
# read in data line by line; check last col to get class # column # 61190, or [-1] contains the label (from 1 -20)


# when new class is found, append to master list


# then within each class, store counts for EACH feature as values in a list. (matrix of values=list of lists)


# results in matrix of 20 rows and 61188 columns
#####################################################

# index 0 should be a dummy value at beginning of each new list
#  as when test data is run it should line up, i.e val for feature 1 is in index 1
#                feat  1 2 3 4 5
#  no 0th label   #  0 0 0 0 0 0
#  1st label      #  0 2 8 9 0 1
#  2nd label      #  0 1 2 1 8 2
#  3rd label      #  0 4 0 2 6  . . .

# log and across each matrix row p(x1....n| y), taking dot product with vector of py's, then sum
# take the largest row val and that is your prediction.
# return index of row and match to dict key or val


###MATHEMATICAL METHODS##################
#TODO:
#P(Y) given by MLE= multinomial? we have 20 class values (figure this out)
# estimate P (X|Y ) using a MAP estimate with the prior distribution Dirichlet(1 + b, ..., 1 + b),
#where b = 1/|V | and V is vocabulary (figure this out)

