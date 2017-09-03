import os
import os.path
import argparse
from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

values = load_h5py(args.data)

valuex = TSNE().fit_transform(values[0])

y = []

for i in range(len(values[1])):
	for j in range(len(values[1][1])):
		if values[1][i][j] == 1:
			y.append(j)

plt.scatter(valuex[:,0], valuex[:,1],c=y)
plt.savefig(args.plots_save_dir)
#matplotlib.get_backend()
plt.show()
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())



 

