import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

training_data = pd.read_csv('xy_train9000_smpl.csv')
test_data = pd.read_csv('xy_test9000_smpl.csv')

traindata = training_data.iloc[:,:-1]
#pd.read_csv('x_train_gr_smpl.csv')

trainlabel = training_data.iloc[:,-1:]
#pd.read_csv('y_train_smpl.csv')

testdata = test_data.iloc[:,:-1]
#pd.read_csv('x_test_gr_smpl.csv')

testlabel = test_data.iloc[:,-1:]
#pd.read_csv('y_test_smpl.csv')


traindata = sklearn.preprocessing.normalize(traindata, norm='l2', axis=1, copy=True, return_norm=True)[0]
testdata = sklearn.preprocessing.normalize(testdata, norm='l2', axis=1, copy=True, return_norm=True)[0]

#x = pd.read_csv('x_train_gr_smpl.csv')
#x_test = pd.read_csv('x_test_gr_smpl.csv')
#y = pd.read_csv('y_train_smpl.csv')
#y_test = pd.read_csv('y_test_smpl.csv')

#x = preprocessing.normalize(x)

#traindata = shuffle(traindata,random_state = 30)
#trainlabel = shuffle(trainlabel.values.ravel(),random_state = 30)
#testdata = shuffle(testdata,random_state = 30)
##testlabel = shuffle(testlabel,random_state = 30)

x_train, x_test, y_train, y_test = train_test_split(traindata, trainlabel.values.ravel(), test_size=0.2, random_state=30)

def svc():

    model = LinearSVC(max_iter= 1000)
    model.fit(x_train,y_train)

    predictions = model.predict(x_test)
    #predictions = cross_val_predict(model, traindata, trainlabel, cv=10)

    print(classification_report(y_test, predictions))
    cnf_matrix = confusion_matrix(y_test, predictions)
    print(cnf_matrix)

    sns.heatmap(cnf_matrix)
    plt.title('Heatmap')
    plt.figure(1)

    calculate_TFandFP(cnf_matrix)

    print(sklearn.metrics.accuracy_score(y_test, predictions))

    try:
        roc = roc_auc_score_multiclass(y_test, predictions)
        print(roc)
    except ValueError:
        pass
    plt.show()

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

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

svc()

