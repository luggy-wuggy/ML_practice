import numpy as np
from operator import itemgetter
from collections import Counter



def run (Xtrain_file, Ytrain_file, test_data_file, pred_file):
	"""
	The function to run your ML algorithm on given datasets,
	generate the predictions and save them into the provided file path
	"""

	## Xtrain_file : string (the path to Xtrain csv file)
	## Ytrain_file : string (the path to Ytrain csv file)
	## test_data_file : string (the path to test data csv file)
	## pred_file : string (the prediction file to be saved by your code)

	### Return a list of modes from a list
	def mode(lst):
		counter = Counter(lst)
		_,val = counter.most_common(1)[0]
		return [x for x,y in counter.items() if y == val]

	K = 8

	x = np.loadtxt(Xtrain_file, dtype = float, delimiter = ',')
	y = np.loadtxt(Ytrain_file, dtype = float, delimiter = ',')
	test = np.loadtxt(test_data_file, dtype = float, delimiter = ',')

	predictions = []

	## KNN classifier ====TRAINING/TESTING =====
	for v in test:

		distances = []

		## Compute distance for every training instances
		for i in range(len(x)):
			euclidan_distance = np.linalg.norm(v - x[i])
			v_class = y[i]
			distances.append( (euclidan_distance, v_class))

		## Sort the distances by closests to farthests
		## Also slice it to focus only on K neigbors
		distances = sorted(distances, key=itemgetter(0))
		distances = distances[0:K]

		modes = mode( [i[1] for i in distances] )

		if len(modes) > 1:

			## In case of multiple modes, 
			## predict the class that is closer distance
			for d in distances:
				if d[1] in modes:
					prediction = d[1]
					break
		else:
			prediction = modes[0]		

		predictions.append(prediction)

	np.savetxt(pred_file, predictions, fmt='%1d', delimiter=",")


	return predictions


def main (Xtrain_file, Ytrain_file, training_size):
	''' 
	split data into training and testing data
	call run with the parameters from training 
	'''

	## 1. Read the CSV files

	x = np.loadtxt(Xtrain_file, dtype = float, delimiter=',')
	y = np.loadtxt(Ytrain_file, dtype = float, delimiter=',')

	## 2. Zip the feature vectors with labels

	whole = np.column_stack((x, y))
	np.random.shuffle(whole)   

	## 3. Split the last 10% of training data for test_data_file

	training_90 = whole[ :int(len(whole)*0.9), :]
	test = whole[int(len(whole)*0.9):, :]

	## 4. Compute the 1%, 2%, 5%, 10%, 20%, and 100% of the training data for report

	training = training_90[ :int(len(training_90) * training_size) , :]

	x_train = training[ :, :len(training[0])-1]
	y_train = training[ :, -1]

	x_test = test[ :,  :len(test[0])-1]
	y_test = test[ :, -1]

	np.savetxt("x_train", x_train, fmt='%1.4f', delimiter=",")
	np.savetxt("y_train", y_train, fmt='%1.4f', delimiter=",")

	np.savetxt("x_test", x_test, fmt='%1.4f', delimiter=",")
	np.savetxt("y_test", y_test, fmt='%1.4f', delimiter=",")

	run ("x_train", "y_train", "x_test", "prediction")

	return 0


def accuracy_test(pred_file, true_file):

	pred = np.loadtxt(pred_file, delimiter = ',')
	true = np.loadtxt(true_file, delimiter = ',')

	corrects = 0
	for i in range(len(pred)):
		if pred[i] == true[i]:
			corrects += 1

	print(pred)
	print(true)
	print("Accuracy: ", corrects/len(pred))




if __name__ == "__main__":

	Xtrain_file = "Xtrain.csv"
	Ytrain_file = "Ytrain.csv"
	training_size = 1.0

	main(Xtrain_file, Ytrain_file, training_size)
	accuracy_test("prediction", "y_test")
	
	




