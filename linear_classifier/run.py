import numpy as np

# This will be the code you will start with, note that the formate of the function 
# run (train_dir,test_dir,pred_file) and saving method can not be changed
# Feel free to change anything in the code

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):


    ## data : data of specific class instances
    ## calculate and return the centroid of the class, by taking the average of all the points
    def centroid(data):
        class_centroid = [0] * len(data[0][0])

        for i in range(len(data)):

            j = 0

            while j<len(class_centroid):
                class_centroid[j] += data[i][0][j]
                j+=1

        class_centroid = list(map(lambda x: x/len(data), class_centroid))


        return class_centroid

    ## datapoint : a N-dimensional datapoint
    ## centroid : a N-dimensional centroid
    ## calculate and return the distance from datapoint to centroid
    ## using L2 distance measures
    def distance(datapoint, centroid):
        total_sum = 0

        for i in range(len(centroid)):
            total_sum += ((centroid[i] - datapoint[i])**2)

        distance = total_sum ** (0.5)

        return distance

    ## datapoint : a N-dimensional datapoint
    ## centroid0,1,2 : a N-dimensional centroid for its respecitive class
    ## Using the nearest-neighbor classifier, predict which centroid is the closest to the datapoint
    ## Return the class prediction that is the closest centroid
    def discriminant_func(datapoint, centroid0, centroid1, centroid2):

        prediction = 0

        c0 = distance(datapoint, centroid0)
        c1 = distance(datapoint, centroid1)
        c2 = distance(datapoint, centroid2)

        if c0 <= c1:
            prediction = 0
            if c0 > c2:
                prediction = 2
        else:
            prediction = 1
            if c1 > c2:
                prediction = 2

        return prediction


    ## load the training data and label,
    ## and pair the data points features with the labels
    training_data = np.loadtxt(train_input_dir,skiprows=0)
    training_data_label = np.loadtxt(train_label_dir,skiprows=0)
    training_data_whole = []

    for i in range(len(training_data)):
        training_data_whole.append( (training_data[i], training_data_label[i]) )


    class_0_instances = []
    class_1_instances = []
    class_2_instances = []
        
    ## place each instances of training data into designted class bucket
    for i in range(len(training_data_whole)):

        if training_data_whole[i][1] == 0.0:
            class_0_instances.append(training_data_whole[i])
        elif training_data_whole[i][1] == 1.0:
            class_1_instances.append(training_data_whole[i])
        else:
            class_2_instances.append(training_data_whole[i])


    centroid0 = centroid(class_0_instances)
    centroid1 = centroid(class_1_instances)
    centroid2 = centroid(class_2_instances)

    testing_data = np.loadtxt(test_input_dir,skiprows=0)
    prediction = []

    for i in range(len(testing_data)):
        prediction.append((discriminant_func(testing_data[i], centroid0, centroid1, centroid2)))

    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

def test (pred_file, true_file):
    pred = np.loadtxt(pred_file,skiprows=0)
    true = np.loadtxt(true_file,skiprows=0)


    ## 3x3 contingency table, representing for number of classification
    a = np.zeros(shape=(3,3))

    for i in range(len(pred)):
        pair = [int(pred[i]), int(true[i])]

        if (pair == [0,0]):
            a[0][0] += 1
        elif (pair == [1,1]):
            a[1][1] += 1
        elif (pair == [2,2]):
            a[2][2] += 1
        elif (pair == [0,1]):
            a[0][1] += 1
        elif (pair == [0,2]):
            a[0][2] += 1
        elif (pair == [1,0]):
            a[1][0] += 1
        elif (pair == [1,2]):
            a[1][2] += 1
        elif (pair == [2,0]):
            a[2][0] += 1
        else:
            a[2][1] += 1

    first_row = a[0][0] + a[1][0] + a[2][0]
    second_row = a[0][1] + a[1][1] + a[2][1]
    third_row = a[0][2] + a[1][2] + a[2][2]
    total = first_row + second_row + third_row

    true_positives = a[0][0] + a[1][1] + a[2][2]

    recall_01 = a[0][0]/(a[0][0]+a[0][1])
    recall_02 = a[0][0]/(a[0][0]+a[0][2])
    recall_12 = a[1][1]/(a[1][1]+a[1][2])
    avg_recall = (recall_01 + recall_02 + recall_12)/3

    precision_01 = a[0][0]/(a[0][0]+a[1][0])
    precision_02 = a[0][0]/(a[0][0]+a[2][0])
    precision_12 = a[1][1]/(a[1][1]+a[2][1])
    avg_precision = (precision_01 + precision_02 + precision_12)/3

    f_denominator = 0.5 * ((1/(avg_recall)+ (1/(avg_precision))))

    f1_score = (1/f_denominator)

    print("total # of instances: ",total)
    print("accuracy: ",true_positives/total)
    print("f1_score: ", f1_score)

    
if __name__ == "__main__":

    train_input_dir = 'training1.txt'
    test_input_dir = 'testing2.txt'


    train_label_dir = 'training1_label.txt'
    test_label_dir = 'testing2_label.txt'

    pred_file = 'pred_file'


    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
    test (pred_file, test_label_dir)








