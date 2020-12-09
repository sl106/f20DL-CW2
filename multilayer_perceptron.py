
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from xlwt import Workbook
simplefilter("ignore", category=ConvergenceWarning)

# The datasets are saved into dataframes. If its the full data set, each set is read individually
# but for the reduced ones, since the labels are included in the datasets, they are separated
training_data = pd.read_csv('xy_train9000_smpl.csv') #reduced training dataset to dataframe
test_data = pd.read_csv('xy_test9000_smpl.csv') #reduced testset dataset to dataframe

traindata = training_data.iloc[:,:-1] #pixels from dataset selected
#traindata =pd.read_csv('x_train_gr_smpl.csv') #full dataset pixels

trainlabel = training_data.iloc[:,-1:] #labels from dataset selected
#trainlabel = pd.read_csv('y_train_smpl.csv') #full dataset labels

testdata = test_data.iloc[:,:-1] #pixels from dataset selected
#testdata = pd.read_csv('x_test_gr_smpl.csv') #full testset pixels

testlabel = test_data.iloc[:,-1:] #labels from dataset selected
#testlabel = pd.read_csv('y_test_smpl.csv') #full testset labels

#the pixels in both datasets with pixels are normalized
traindata = sklearn.preprocessing.normalize(traindata, norm='l2', axis=1, copy=True, return_norm=True)[0]
testdata = sklearn.preprocessing.normalize(testdata, norm='l2', axis=1, copy=True, return_norm=True)[0]

#A class is created to save the parameters of each of the tests done
class Test:
    def __init__(self,type,accuracy,layers,neurons_per_layer,learning_rate,momentum,state):
        self.accuracy = accuracy
        self.neurons_per_layer = neurons_per_layer
        self.layers = layers
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.state = state
        self.type = type

#The method for the multilayer perceptron, returns the accuracies for two tests: training and testset or training and cross validation
def Multilayer_Perceptron(layers,neurons,lr,momentum,state,figure):
    # To fit the layers and neurons a tupple is used
    structure = ()

    for i in range(0, layers):
        structure = structure + (neurons,)

    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(structure), random_state=state, max_iter=1000,learning_rate_init=lr, momentum=momentum)

    print('Layers: ', layers, 'neurons: ', neurons, 'learning rate: ', lr, 'momentum: ', momentum, 'seed: ', state)

    # Different methods for training, testset and cross validation are called here and their accuracies are stored.
    # The methods return a confusion matrix which is used to create a heatmap
    print('For training:')
    training_acc, cnf_train = training(model)
    plt.figure(figure)
    sns.heatmap(cnf_train)
    plt.title('Heatmap Confusion Matrix. Training' + ' L: ' + str(layers) + ' N: ' + str(neurons) + 'LR: ' + str(lr) + 'Momentum:' + str(momentum) + 'Seed: ' + str(state))

    print('For Crossval:')
    test_acc, cnf_test = cross_val(model)
    plt.figure(figure + 1)
    sns.heatmap(cnf_test)
    plt.title('Heatmap Confusion Matrix. Cross validation' + ' L: ' + str(layers) + ' N: ' + str(neurons) + 'LR: ' + str(lr) + 'Momentum:' + str(momentum) + 'Seed: ' + str(state))

    return training_acc,test_acc

#The data is divided into a proportion of 80 for training and 20 for testing.
#It passes the labels for the test and the predictions to inform method which will print the results
def training(model):

    x_train, x_test, y_train, y_test = train_test_split(traindata, trainlabel.values.ravel(), test_size=0.2)
    model.fit(x_train,y_train)

    predictions = model.predict(x_test)
    return inform(y_test,predictions)

#The testing dataset is used to make predictions
#It passes the labels for the test and the predictions to inform method which will print the results
def testset(model):

    model.fit(traindata,trainlabel.values.ravel())

    predictions = model.predict(testdata)
    return inform(testlabel,predictions)

#The 10-fold cross validation is done here
#It passes the labels for the test and the predictions to inform method which will print the results
def cross_val(model):

    raveled = trainlabel.values.ravel()
    model.fit(traindata, raveled)

    predictions = cross_val_predict(model, traindata, raveled, cv=10)
    return inform(raveled,predictions)

#To calcute the confusion matrix, the classification report,accuracy, TF and FP and the ROC areas the different methods are called here
#The accuracy and the confusion matrix are returned
def inform(actual,predictions):

    cnf_matrix = confusion_matrix(actual, predictions)
    print(cnf_matrix)

    print(classification_report(actual, predictions))

    calculate_TFandFP(cnf_matrix)

    accuracy = sklearn.metrics.accuracy_score(actual, predictions)

    try:
        roc = roc_auc_score_multiclass(actual, predictions)
        print(roc)
    except ValueError:
        pass

    return accuracy,cnf_matrix

#The TF and FP are calculated, returning the average TF and FP
def calculate_TFandFP(cnf_matrix):

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    #TNR = TN / (TN + FP)
    # Precision or positive predictive value
    #PPV = TP / (TP + FP)
    # Negative predictive value
    #NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    #FNR = FN / (TP + FN)
    # False discovery rate
    #FDR = FP / (TP + FP)
    # Overall accuracy
    #ACC = (TP + TN) / (TP + FP + FN + TN)
    print("TPR: ", Average(TPR), "FPR: ", Average(FPR))

def Average(lst):
    return sum(lst) / len(lst)

#Afsan Abdulali Gujarati
#https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
#This method calculates the ROC areas for each class
def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

#Finds the test with the best accuracy so far
def calculate_best(tests):
    best = None
    for test in tests:
        if best == None or test.accuracy > best.accuracy:
            best = test
    return best

#Main method where all the parameters are introduced and the graphs are plotted
def run_experiments():
    
    figure = 1
    tests = []

    acc_train = []
    acc_test = []

    layers = [1,2,3]
    neurons = [1,5,11,15,25,29]

    #The layers and neurons are varied, the parameters used and the resulting accuracies are stored in a Test class
    for i in range(0,len(layers)): #1,4
        for n in range(0, len(neurons)): #20,30,2
            t1,t2 = Multilayer_Perceptron(layers[i],neurons[n],0.0001,0.9,1,figure)
            acc_train.append(t1)
            acc_test.append(t2)
            tests.append(Test('Training',t1,layers[i],neurons[n],0.001,0.9,1))
            tests.append(Test('Testset',t2,layers[i],neurons[n],0.001,0.9,1))
            figure = figure + 2

    #3D graphs showing the variations of accuracies with the layers and neurons are created
    print('Training')
    plot_3D(acc_train, figure)
    figure = figure + 1
    print('Tests')
    plot_3D(acc_test, figure)
    figure = figure + 1

    #Looks for the best test so far so its layers and neurons parameters can be used on the following tests, thus finding the best parameters to classify the pictures
    best1 = calculate_best(tests)

    lr_acc_train = []
    lr_acc_test = []
    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    # The learning rate is varied, the parameters used and the resulting accuracies are stored in a Test class
    for j in range(0,len(learning_rates)):
        t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1, figure)

        lr_acc_train.append(t1)
        lr_acc_test.append(t2)
        tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1))
        tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1))
        figure = figure + 2

    #A graph with the learnong rates used and the accuracies for training and testing or training and corss validation
    plot_2D(learning_rates,lr_acc_train,lr_acc_test, figure)
    figure = figure + 1

    # Looks for the best test so far so its learning rate parameter can be used on the following tests
    best1 = calculate_best(tests)

    mm_acc_train = []
    mm_acc_test = []
    momentum = [0.2, 0.4, 0.6, 0.8]

    # The momentum is varied, the parameters used and the resulting accuracies are stored in a Test class
    for k in np.arange(0.2,1.0,0.2):
        t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1, figure)
        mm_acc_train.append(t1)
        mm_acc_test.append(t2)
        tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1))
        tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1))
        figure = figure + 2

    # A graph with the momentum used and the accuracies for training and testing or training and corss validation
    plot_2D(momentum, mm_acc_train, mm_acc_test, figure)
    figure = figure + 1

    # Looks for the best test so far so its momentum parameter can be used on the following tests
    best1 = calculate_best(tests)

    seed_acc_train = []
    seed_acc_test = []
    seeds = [5, 15, 25, 30, 45] #5, 10, 15, 20, 25, 30, 35, 40, 45

    # The seed is varied, the parameters used and the resulting accuracies are stored in a Test class
    for s in range(0,len(seeds)):
        t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, seeds[s], figure)
        seed_acc_train.append(t1)
        seed_acc_test.append(t2)
        tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, seeds[s]))
        tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, seeds[s]))
        figure = figure + 2

    # A graph with the seeds used and the accuracies for training and testing or training and corss validation
    plot_2D(seeds, seed_acc_train, seed_acc_test, figure)

    # Looks for the best test so it can be printed
    best1 = calculate_best(tests)

    #The accuracies of the tests are shown
    for test in tests:
        print('Class type:',test.type,'Layers: ',test.layers, 'neurons per layer: ', test.neurons_per_layer, 'learning rate: ', test.learning_rate, 'momemtum: ', test.momentum, 'seed: ', test.state, 'accuracy. ', test.accuracy)

    #The best accuracy obtained from tests is printed
    print('The best was:')
    print('Class type:', best1.type, 'Layers: ', best1.layers, 'neurons per layer: ', best1.neurons_per_layer, 'learning rate: ', best1.learning_rate, 'momemtum: ', best1.momentum, 'seed: ', best1.state, 'accuracy. ', best1.accuracy)

    #All the the results are saved to an excel file
    to_excel(tests)

#The 3D graph is created with the layers, neurons and accuracies abtained from the tests
#In order to make it easier to see the accuracies, the smallest accuracy is found and the rest are plotted with respect to it
#So cubes are not joined, the layers and neuron numbers are slightly reduced
def plot_3D(accuracy, pic):

    fig = plt.figure()
    axl = plt.axes(projection='3d')

    x = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3] #1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    y = [1,5,11,15,25,29,1,5,11,15,25,29,1,5,11,15,25,29]#1,5,7,9,11,13,15,17,19,21,23,25,27,29,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29
    z = []
    dy = []
    dx = []
    dz = []

    min = None
    for i in accuracy:
        if min == None or min > i:
            min = i

    accuracy_normalized = []
    for i in range(0, len(accuracy)):
        accuracy_normalized.append((accuracy[i] - min) * 100)  # percentage

    for i in range(0, len(accuracy)):
        x[i] = x[i] - 0.25
        y[i] = y[i] - 0.25
        dz.append(accuracy_normalized[i])
        z.append(0)
        dx.append(0.5)
        dy.append(0.5)

    plt.title('Precision VS Layers VS Neurons')
    axl.set_xlabel('Layers')
    axl.set_ylabel('Neurons per layer')
    axl.set_zlabel('Percentage change in accuracy')
    axl.bar3d(x, y, z, dx, dy, dz)
    plt.figure(pic)

#Graphs for the training and testing or training and crosvalidation are created with the parameters used
def plot_2D(values,train_acc,test_acc, pic):

    plt.figure(pic)

    plt.subplot(211)
    plt.plot(values, train_acc)
    plt.title('train accuracy')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(values, test_acc)
    plt.title('test accuracy')
    plt.grid(True)

#This method writes the data gathered from the tests into an excel file. 
def to_excel(array):

    wb = Workbook()
    pointer=0
    sheet1 = wb.add_sheet('Results')
    sheet1.write(0, 0, 'Layers')
    sheet1.write(0, 1, 'Neurons')
    sheet1.write(0, 2, 'Learning Rate')
    sheet1.write(0, 3, 'Momentum')
    sheet1.write(0, 4, 'Seed')
    sheet1.write(0, 5, 'Accuracy Training')
    sheet1.write(0, 6, 'Accuracy Test')

    # add_sheet is used to create sheet.
    for i in range(0,len(array),2):

        pointer=pointer+1
        record = array[i]
        nextRecord = array[i+1]
        sheet1.write(pointer, 0, record.layers)
        sheet1.write(pointer, 1, record.neurons_per_layer)
        sheet1.write(pointer, 2, record.learning_rate)
        sheet1.write(pointer, 3, record.momentum)
        sheet1.write(pointer, 4, record.state)
        sheet1.write(pointer, 5, record.accuracy)
        sheet1.write(pointer, 6, nextRecord.accuracy)
    wb.save('data_for_9000.xls')

run_experiments()
plt.show()
