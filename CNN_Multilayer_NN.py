import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
import sklearn
import seaborn as sns
import xlwt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from tensorflow.python.keras.utils.np_utils import to_categorical
from xlwt import Workbook
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
simplefilter("ignore", category=ConvergenceWarning)

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 48, 48, 1
loss_function = categorical_crossentropy
no_classes = 10
no_epochs = 10
optimizer = Adam()
validation_split = 0.2
verbosity = 1
# Determine shape of the data
input_shape = (img_width, img_height,img_num_channels)

np.set_printoptions(threshold= sys.maxsize)
'''
traindata = pd.read_csv('Train/x_train_gr_smpl.csv')
trainlabel = pd.read_csv('Train/y_train_smpl.csv')
testdata = pd.read_csv('Test/x_test_gr_smpl.csv')
testlabel = pd.read_csv('Test/y_test_smpl.csv')
'''
training_data = pd.read_csv('Train/xy_train9000_smpl.csv')
test_data = pd.read_csv('Test/xy_test9000_smpl.csv')


traindata = training_data.iloc[:,:-1]
print(traindata.shape)
trainlabel = training_data.iloc[:,-1:]
print(trainlabel.shape)
testdata = test_data.iloc[:,:-1]
print(testdata.shape)
testlabel = test_data.iloc[:,-1:]
print(testlabel.shape)

traindata = sklearn.preprocessing.normalize(traindata, norm='l2', axis=1, copy=True, return_norm=True)[0]
testdata = sklearn.preprocessing.normalize(testdata, norm='l2', axis=1, copy=True, return_norm=True)[0]

class Test:
    def __init__(self,type,accuracy,layers,neurons_per_layer,momentum,learning_rate,state,kernel):
        self.accuracy = accuracy
        self.neurons_per_layer = neurons_per_layer
        self.layers = layers
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.state = state
        self.type = type
        self.kernel = kernel

def Multilayer_Perceptron(layers,neurons,lr,momentum,state,figure):

    structure = ()

    for i in range(0, layers):
        structure = structure + (neurons,)

    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(structure), random_state=state, max_iter=1000,learning_rate_init=lr, momentum=momentum)

    print('Layers: ', layers, 'neurons: ', neurons, 'learning rate: ', lr, 'momentum: ', momentum, 'seed: ', state)
    print('For training:')
    training_acc, cnf_train = training(model)
    plt.figure(figure)
    sns.heatmap(cnf_train)
    plt.title('Heatmap Confusion Matrix. Training' + ' L: ' + str(layers) + ' N: ' + str(neurons) + 'LR: ' + str(lr) + 'Momentum:' + str(momentum) + 'Seed: ' + str(state))

    print('For testset:')
    test_acc, cnf_test = testset(model)
    plt.figure(figure + 1)
    sns.heatmap(cnf_test)
    plt.title('Heatmap Confusion Matrix. Testset' + ' L: ' + str(layers) + ' N: ' + str(neurons) + 'LR: ' + str(lr) + 'Momentum:' + str(momentum) + 'Seed: ' + str(state))

    return training_acc,test_acc

def training(model):

    x_train, x_test, y_train, y_test = train_test_split(traindata, trainlabel.values.ravel(), test_size=0.2)

    model.fit(x_train,y_train)

    predictions = model.predict(x_test)
    return inform(y_test,predictions)

def testset(model):

    model.fit(traindata,trainlabel.values.ravel())

    predictions = model.predict(testdata)
    return inform(testlabel,predictions)

def convolutionalNeuralNet(n_layers,neurons,state,figure,kernel): # as the kernel matrix is always square one parameter serve for both
    model = Sequential()
    print(input_shape)
    model.add(Conv2D(44, kernel_size=kernel, padding='same', activation='relu', input_shape=input_shape)) #defining the layers of the CNN
    model.add(MaxPool2D())
    model.add(Conv2D(64, kernel_size=kernel, padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(86, kernel_size=kernel, padding='same', activation='relu'))
    model.add(MaxPool2D(padding='same'))
    #model.add(Conv2D(98, kernel_size=kernel, padding='same', activation='relu'))
    #model.add(MaxPool2D(padding='same'))
    #model.add(Conv2D(122, kernel_size=kernel, padding='same', activation='relu'))
    #model.add(MaxPool2D(padding='same'))
    #option to bear in mind
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    traindataNP = np.asarray(traindata)
    traindataNP = traindataNP.reshape((traindata.shape[0], 48, 48, 1))
    trainlabelNP = np.asarray(trainlabel)
    trainlabelNP = trainlabelNP.reshape((trainlabel.shape[0], 1))
    x_train, x_test, y_train, y_test = train_test_split(traindataNP, trainlabelNP, test_size=0.2)
    label_encoder = LabelEncoder() #Encode the labels for the reduced dataset which solutions include and s before the class number
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, 10) #changing the class to an array of 10 places with all 0 except for the index of the class who which the sample belong
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    model.fit(x_train, y_train, epochs=no_epochs)
    y_test = label_encoder.fit_transform(y_test)
    localTestLabel = label_encoder.fit_transform(testlabel)
    model.summary()
    # Compile the model
    #model.compile(loss=loss_function,
    #              optimizer=optimizer,
    #              metrics=['accuracy'])

    print('Layers: ', n_layers, 'neurons: ', neurons, 'kernel: ', kernel)
    print('For training:')
    training_acc, cnf_train = training_conv(model,x_test,y_test)
    plt.figure(figure)
    sns.heatmap(cnf_train)
    plt.title('Heatmap Confusion Matrix. Training' + ' L: ' + str(n_layers) + ' N: ' + str(neurons) + 'K: ' + str(
        kernel) )

    print('For testset:')
    test_acc, cnf_test = testset_conv(model,localTestLabel)
    plt.figure(figure + 1)
    sns.heatmap(cnf_test)
    plt.title('Heatmap Confusion Matrix. Testset' + ' L: ' + str(n_layers) + ' N: ' + str(neurons) + 'K: ' + str(
        kernel))

    return training_acc, test_acc



def training_conv(model,x_test,y_test):

    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    #predicted_classes = np.asarray(predicted_classes)
    #print(predicted_classes)
    #predicted_classes=predicted_classes.reshape(predicted_classes.shape[0], 1)
    #print(predicted_classes)
    return inform(y_test,predicted_classes)


def testset_conv(model,localtTestLabel):
    testdataNP = np.asarray(testdata)
    testdataNP = testdataNP.reshape((testdata.shape[0], 48, 48, 1))
    predictions = model.predict(testdataNP)
    predicted_classes = np.argmax(predictions, axis=1)
    return inform(localtTestLabel,predicted_classes)

def inform(actual,predictions):
    cnf_matrix = confusion_matrix(actual, predictions)
    print(cnf_matrix)
    #sns.heatmap(cnf_matrix)

    print(classification_report(actual, predictions))

    calculate_TFandFP(cnf_matrix)

    accuracy = sklearn.metrics.accuracy_score(actual, predictions)

    #plt.title('Heatmap Confusion Matrix.' +' L: ' + str(n_layers) + ' N: ' + str(n_neurons))
    #plt.figure(run)
    try:
        print("")
        #roc = roc_auc_score_multiclass(tuple(actual),tuple(predictions))
        #print(roc)
    except ValueError:
        pass

    return accuracy,cnf_matrix

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
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("TPR: ", Average(TPR), "FPR: ", Average(FPR))

def Average(lst):
    return sum(lst) / len(lst)

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

def calculate_best(tests):
    best = None
    for test in tests:
        if best == None or test.accuracy > best.accuracy:
            best = test
    return best

def run_experiments():
    figure = 1
    tests = []
    type = 1 #convolutional
    acc_train = []
    acc_test = []

    for i in range(2,3): #1,4
        t1=None
        t2=None
        for n in range(30, 31, 2): #20,30,2
            for k in range(2, 10):  # 20,30,2
                if type==0:
                    t1,t2 = Multilayer_Perceptron(i,n,0.0001,0.9,1,figure)
                if type==1:
                    t1, t2 = convolutionalNeuralNet(i, n, 1, figure, k)
                acc_train.append(t1)
                acc_test.append(t2)
                if type==0:
                    tests.append(Test('Training', t1, 1, 10, 0.001, 0.9, 3))
                    tests.append(Test('Testset', t2, 1, 10, 0.001, 0.9, 3))
                if type==1:
                    tests.append(Test('Training', t1, 1, 10, 1,1,1, k))
                    tests.append(Test('Testset', t2, 1, 10,1,1,1, k))

                figure = figure + 2

    print('Training')
    #plot_3D(acc_train, figure)
    figure = figure + 1
    print('Tests')
    #plot_3D(acc_test, figure)
    figure = figure + 1

    best1 = calculate_best(tests)

    lr_acc_train = []
    lr_acc_test = []
    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    if type == 0:
        for j in range(0,len(learning_rates)):
            t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1, figure)
            lr_acc_train.append(t1)
            lr_acc_test.append(t2)
            tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1))
            tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, learning_rates[j], 0.9, 1))
            figure = figure + 2

        plot_2D(learning_rates,lr_acc_train,lr_acc_test, figure)
        figure = figure + 1

        best1 = calculate_best(tests)

        mm_acc_train = []
        mm_acc_test = []
        momentum = [0.2, 0.4, 0.6, 0.8]

        for k in np.arange(0.2,1.0,0.2):
            t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1, figure)
            mm_acc_train.append(t1)
            mm_acc_test.append(t2)
            tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1 ))
            tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, best1.learning_rate, k, 1 ))
            figure = figure + 2

        plot_2D(momentum, mm_acc_train, mm_acc_test, figure)
        figure = figure + 1

        best1 = calculate_best(tests)

        seed_acc_train = []
        seed_acc_test = []
        seeds = [5, 10, 15, 20, 25, 30, 35, 40, 45]

        for s in range(5,50,5):
            t1, t2 = Multilayer_Perceptron(best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, s, figure)
            seed_acc_train.append(t1)
            seed_acc_test.append(t2)
            tests.append(Test('Training', t1, best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, s))
            tests.append(Test('Testset', t2, best1.layers, best1.neurons_per_layer, best1.learning_rate, best1.momentum, s))
            figure = figure + 2
        plot_2D(seeds, seed_acc_train, seed_acc_test, figure)

        best1 = calculate_best(tests)

        for test in tests:
            print('Class type:',test.type,'Layers: ',test.layers, 'neurons per layer: ', test.neurons_per_layer, 'learning rate: ', test.learning_rate, 'momemtum: ', test.momentum, 'seed: ', test.state, 'accuracy. ', test.accuracy)

        print('The best was:')
        print('Class type:', best1.type, 'Layers: ', best1.layers, 'neurons per layer: ', best1.neurons_per_layer, 'learning rate: ', best1.learning_rate, 'momemtum: ', best1.momentum, 'seed: ', best1.state, 'accuracy. ', best1.accuracy)
    to_excel(tests, type)

def plot_3D(accuracy, pic):

    fig = plt.figure()
    axl = plt.axes(projection='3d')

    x = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    y = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
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

def to_excel(array,type):
    if type==0:
        wb = Workbook()
        pointer = 0
        sheet1 = wb.add_sheet('Results')
        sheet1.write(0, 0, 'Layers')
        sheet1.write(0, 1, 'Neurons')
        sheet1.write(0, 2, 'Learning Rate')
        sheet1.write(0, 3, 'Momentum')
        sheet1.write(0, 4, 'Seed')
        sheet1.write(0, 5, 'Accuracy Training')
        sheet1.write(0, 6, 'Accuracy Test')

        # add_sheet is used to create sheet.
        for i in range(0, len(array), 2):
            print(pointer)
            pointer = pointer + 1
            record = array[i]
            nextRecord = array[i + 1]
            sheet1.write(pointer, 0, record.layers)
            sheet1.write(pointer, 1, record.neurons_per_layer)
            sheet1.write(pointer, 2, record.learning_rate)
            sheet1.write(pointer, 3, record.momentum)
            sheet1.write(pointer, 4, record.state)
            sheet1.write(pointer, 5, record.accuracy)
            sheet1.write(pointer, 6, nextRecord.accuracy)
        wb.save('data_multilayer.xls')
    if type == 1:
        wb = Workbook()
        pointer = 0
        sheet1 = wb.add_sheet('Results')
        sheet1.write(0, 0, 'Layers')
        sheet1.write(0, 1, 'Neurons')
        sheet1.write(0, 2, 'Kernel')
        sheet1.write(0, 3, 'Accuracy Training')
        sheet1.write(0, 4, 'Accuracy Test')

        # add_sheet is used to create sheet.
        for i in range(0, len(array), 2):
            print(pointer)
            pointer = pointer + 1
            record = array[i]
            nextRecord = array[i + 1]
            sheet1.write(pointer, 0, record.layers)
            sheet1.write(pointer, 1, record.neurons_per_layer)
            sheet1.write(pointer, 2, record.kernel)
            sheet1.write(pointer, 3, record.accuracy)
            sheet1.write(pointer, 4, nextRecord.accuracy)
        wb.save('data_variation_9000_kernel.xls')


run_experiments()
plt.show()
