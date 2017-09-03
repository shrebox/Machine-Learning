import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

values = load_h5py(args.test_data)

y = []

for i in range(len(values[1])):
	for j in range(len(values[1][1])):
		if values[1][i][j] == 1:
			y.append(j)

x = values[0]

if args.model_name == 'GaussianNB':
	model = joblib.load(args.weights_path)
	prediction = model.predict(x)
	file = open(args.output_preds_file,'w')
	for i in range(len(prediction)):
		print prediction[i]
		file.write(str(prediction[i])+"\n")
	file.close()
	#np.savetxt(args.output_preds_file,prediction, delimiter=',')
	#print np.genfromtxt(args.output_preds_file, delimiter=',')

elif args.model_name == 'LogisticRegression':
	model = joblib.load(args.weights_path)
	prediction = model.predict(x)
	file = open(args.output_preds_file,'w')
	for i in range(len(prediction)):
		print prediction[i]
		file.write(str(prediction[i])+"\n")
	file.close()
	#np.savetxt(args.output_preds_file,prediction, delimiter=',')
	#print np.genfromtxt(args.output_preds_file, delimiter=',')

elif args.model_name == 'DecisionTreeClassifier':
	model = joblib.load(args.weights_path)
	prediction = model.predict(x)
	file = open(args.output_preds_file,'w')
	for i in range(len(prediction)):
		print prediction[i]
		file.write(str(prediction[i])+"\n")
	file.close()
	#np.savetxt(args.output_preds_file,prediction, delimiter=',')
	#print np.genfromtxt(args.output_preds_file, delimiter=',')
	# load the model

	# model = DecisionTreeClassifier(  ...  )

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
else:
	raise Exception("Invald Model name")
