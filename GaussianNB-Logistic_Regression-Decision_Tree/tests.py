import numpy as np
# from Models.GaussianNB import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

for model in  [ GaussianNB(), LogisticRegression(), DecisionTreeClassifier()] : 
	model.fit(X, Y)
	assert (model.predict([[-0.8, -1]]))[0] == 1


print "If you can see this on stdout, all tests have been passed :)"
