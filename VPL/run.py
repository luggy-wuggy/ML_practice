import numpy as np


def run (Xtrain_file, Ytrain_file, test_data_file, pred_file):
	"""
	The function to run your ML algorithm on given datasets,
	generate the predictions and save them into the provided file path
	"""

	## Xtrain_file : string (the path to Xtrain csv file)
	## Ytrain_file : string (the path to Ytrain csv file)
	## test_data_file : string (the path to test data csv file)
	## pred_file : string (the prediction file to be saved by your code)

	x = np.genfromtxt(Xtrain_file, delimiter=',')
	y = np.genfromtxt(Ytrain_file, delimiter=',')

	test = np.genfromtxt(test_data_file, delimiter=',')

	## Translate all label instances of 0 to -1
	for n, i in enumerate(y):
		if i == 0:
			y[n] = -1
			
	epoch = 5
	w = [ np.zeros_like(x[0]) ]
	c = [0]
	k = 0
	
	## ===Voted Perceptron TRAINING===
	while (epoch != 0):

		for i in range(len(x)):

			value = np.dot(w[k], x[i])

			pred = 1 if (value > 0) else -1

			if (pred == y[i]):
				c[k] += 1
			else:
				dot = np.dot(y[i],x[i])
				new_w = np.array( [a + b for a, b in zip(dot, w[k])] )
				w.append(new_w)
				c.append(1)
				k += 1
		
		epoch -= 1

	# ===Voted Perceptron TESTING===
	prediction = []

	for new_x in test:
		total_sum = 0

		for k in range(len(w)):

			value = np.dot(w[k], new_x)

			m = 1 if value > 0 else -1
			weight = m * c[k]
			total_sum += weight

		final_prediction = 1 if total_sum > 0 else 0
		prediction.append(final_prediction)

	np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

	return 0

def main (Xtrain_file, Ytrain_file, training_size):
	''' 
	split data into training and testing data
	call run with the parameters from training 
	'''

	## 1. Read the CSV files

	x = np.loadtxt(Xtrain_file, dtype = int, delimiter=',')
	y = np.loadtxt(Ytrain_file, dtype = int, delimiter=',')


	## 2. Zip the feature vectors with labels

	whole = np.column_stack((x, y))

	## 3. Split the last 10% of training data for test_data_file

	training_90 = whole[ :int(len(whole)*0.9), :]
	test = whole[int(len(whole)*0.9):, :]

	## 4. Compute the training_size of the training data for report

	training = training_90[ :int(len(training_90) * training_size) , :]

	x_train = training[ :, :len(training[0])-1]
	y_train = training[ :, -1]

	x_test = test[ :,  :len(test[0])-1]
	y_test = test[ :, -1]

	np.savetxt("x_train", x_train, fmt='%1d', delimiter=",")
	np.savetxt("y_train", y_train, fmt='%1d', delimiter=",")

	np.savetxt("x_test", x_test, fmt='%1d', delimiter=",")
	np.savetxt("y_test", y_test, fmt='%1d', delimiter=",")

	run ("x_train", "y_train", "x_test", "prediction")
	
	return 0


def accuracy_test(pred_file, true_file):

	pred = np.loadtxt(pred_file,skiprows=0)
	true = np.loadtxt(true_file,skiprows=0)


    ## 2x2 contingency table, representing for number of classification
	a = np.zeros(shape=(2,2))

	for i in range(len(pred)):
	    pair = [int(pred[i]), int(true[i])]

	    if (pair == [0,0]):
	        a[0][0] += 1
	    elif (pair == [1,1]):
	        a[1][1] += 1
	    elif (pair == [0,1]):
	        a[0][1] += 1
	    else:
	        a[1][0] += 1
	   
	first_row = a[0][0] + a[1][0]
	second_row = a[0][1] + a[1][1]

	first_column = a[0][0] + a[0][1]
	second_column = a[1][0] + a[1][1]

	total = first_row + second_row

	true_positives = a[0][0] + a[1][1]
	precision = true_positives/first_row
	recall = true_positives/first_column

	accuracy = true_positives/total
	f1_score = (2 * precision * recall)/(precision + recall)

	print("total # of instances: ",total)
	print("accuracy: ",true_positives/total)
	print("f1_score: ", f1_score)

	return 0


if __name__ == "__main__":

	Xtrain_file = "Xtrain.csv"
	Ytrain_file = "Ytrain.csv"
	training_size = 1.0
	main(Xtrain_file, Ytrain_file, training_size)
	accuracy_test("prediction", "y_test")


