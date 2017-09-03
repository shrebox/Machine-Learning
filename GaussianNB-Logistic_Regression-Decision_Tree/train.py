import os
import os.path
import argparse
import h5py
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

def kfold_cv(items, k):

    slices = []

    for i in xrange(k):
    	slices.append(items[i::k])

    for i in xrange(k):
        validation = slices[i]
        training = []
        for s in slices:
        	if s is not validation:
        		for item in s:
        			training.append(item)
        yield training, validation

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# Preprocess data and split it

args = parser.parse_args()

values = load_h5py(args.train_data)

y = []

for i in range(len(values[1])):
	for j in range(len(values[1][1])):
		if values[1][i][j] == 1:
			y.append(j)

x = values[0]
#xshort = len(y)/4


# Train the models

if args.model_name == 'GaussianNB':
	count = 0
	sumy = 0
	model = GaussianNB()	

	for i in range(2,11):
		if len(x)%i == 0:
			for trainingx, validationx in kfold_cv(x,i):
				for testx,testy in kfold_cv(y,i):
					model.fit(trainingx,testx)
					sumy += model.score(validationx,testy)
					count += 1
			print "Accuracy of GNB: " + str(sumy/count) + " for K = " + str(i)
			#print count
		else:
			pass

	joblib.dump(model,args.weights_path)

elif args.model_name == 'LogisticRegression':
	count = 0
	sumy = 0
	accuracy = 0
	accl = []
	oppa = []
	countarr = []
	minval = 9999
	countval = 0
	optimal_parameters = {}
	model = LogisticRegression()	

	#using grid_search

	param_grid = {'C': [0.1, 1, 10],'penalty':['l2'],'max_iter':[10,100,500],'verbose':[0,5],'solver':['sag']}

	for cval in range(len(param_grid['C'])):
		for penval in range(len(param_grid['penalty'])):
			for mival in range(len(param_grid['max_iter'])):
				for verval in range(len(param_grid['verbose'])):
					for sol in range(len(param_grid['solver'])):
						model = LogisticRegression(C=param_grid['C'][cval],penalty=param_grid['penalty'][penval],max_iter=param_grid['max_iter'][mival],verbose=param_grid
							['verbose'][verval],solver=param_grid['solver'][sol])
						slicvalx = len(x)/4
						nx = x[0:slicvalx,:]
						for trainingx, validationx in kfold_cv(nx,2):
							for testx,testy in kfold_cv(y[0:slicvalx],2):
								model.fit(trainingx,testx)
								sumy += model.score(validationx,testy)
								count += 1
						accl.append(sumy/count)
						oppa.append({'C':param_grid['C'][cval],'penalty':param_grid['penalty'][penval],'max_iter': param_grid['max_iter'][mival],'verbose': param_grid['verbose'][verval],'solver':param_grid['solver'][sol]})
						countval+=1
						countarr.append(countval)
						if(sumy/count<minval):
							minval = sumy/count
						if(sumy/count>accuracy):
							accuracy = sumy/count
							optimal_parameters = {'C':[param_grid['C'][cval]],'penalty':[param_grid['penalty'][penval]],'max_iter': [param_grid['max_iter'][mival]],'verbose': [param_grid['verbose'][verval]],'solver': [param_grid['solver'][sol]]}
	
	#plotting the values
	plt.xlabel('#Parameter Combinations')
	plt.ylabel('Accuracy')
	plt.scatter(countarr,accl)
	plt.savefig(args.plots_save_dir)
	plt.show()
	model = LogisticRegression(C=optimal_parameters['C'][0],penalty=optimal_parameters['penalty'][0],max_iter=optimal_parameters['max_iter'][0],verbose=optimal_parameters['verbose'][0],solver=optimal_parameters['solver'][0])
	model.fit(x,y)
	joblib.dump(model,args.weights_path)
	print "Avg. Accuracy of LR: " + str(accuracy)
	print optimal_parameters

elif args.model_name == 'DecisionTreeClassifier':
	
	# define the grid here
	#parameters={'min_samples_split' : range(1,500,20),'max_depth': range(1,20,2)}
	count = 0
	sumy = 0
	accuracy = 0
	accl = []
	oppa = []
	countarr = []
	minval = 9999
	countval = 0
	optimal_parameters = {}
	model = DecisionTreeClassifier(random_state=0)
	param_grid={'min_samples_split': [2,5,10],'max_depth': [None,5,10],'min_samples_leaf': [1,5,10],'max_features':[None]}

	for cval in range(len(param_grid['min_samples_split'])):
		for penval in range(len(param_grid['max_depth'])):
			for mival in range(len(param_grid['min_samples_leaf'])):
				for verval in range(len(param_grid['max_features'])):
					model = DecisionTreeClassifier(max_depth=param_grid['max_depth'][penval],min_samples_split=param_grid['min_samples_split'][cval],min_samples_leaf=param_grid['min_samples_leaf'][mival],max_features=param_grid['max_features'][verval])
					slicvalx = len(x)/10
					nx = x[0:slicvalx,:]
					for trainingx, validationx in kfold_cv(nx,2):
						for testx,testy in kfold_cv(y[0:slicvalx],2):
							model.fit(trainingx,testx)
							sumy += model.score(validationx,testy)
							count += 1
					accl.append(sumy/count)
					oppa.append({'min_samples_split':param_grid['min_samples_split'][cval],'max_depth':param_grid['max_depth'][penval],'min_samples_leaf': param_grid['min_samples_leaf'][mival],'max_features': param_grid['max_features'][verval]})
					countval+=1
					countarr.append(countval)
					if(sumy/count<minval):
						minval = sumy/count
					if(sumy/count>accuracy):
						accuracy = sumy/count
						optimal_parameters = {'min_samples_split':[param_grid['min_samples_split'][cval]],'max_depth':[param_grid['max_depth'][penval]],'min_samples_leaf': [param_grid['min_samples_leaf'][mival]],'max_features': [param_grid['max_features'][verval]]}
	#plt.ylim([minval,accuracy])
	plt.xlabel('#Parameter Combinations')
	plt.ylabel('Accuracy')
	plt.scatter(countarr,accl)
	plt.savefig(args.plots_save_dir)
	plt.show()
	model = DecisionTreeClassifier(max_depth=optimal_parameters['max_depth'][0],min_samples_split=optimal_parameters['min_samples_split'][0],min_samples_leaf=optimal_parameters['min_samples_leaf'][0],max_features=optimal_parameters['max_features'][0])
	model.fit(x,y)
	joblib.dump(model,args.weights_path)
	print "Avg. Accuracy of DTC: " + str(accuracy)
	print optimal_parameters

	# save the best model and print the results
else:
	raise Exception("Invald Model name")

#References: http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/

