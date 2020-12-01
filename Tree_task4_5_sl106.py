import sys
assert sys.version_info >= (3, 5)
import pandas as pd
from sklearn.model_selection import train_test_split


#read input data
X = pd.read_csv("resource/train/x_train_gr_smpl.csv")
y = pd.read_csv("resource/train/y_train_smpl.csv")
# testSet = pd.read_csv("resource/newData/xy_test_smpl.csv")

#split out 4000/9000 instances from training data set
#.4127541017%: 4000/9691
#.9286967289, 9000/9691
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9286967289, stratify=y, random_state=17)

#convert slices to dataframes then insert y to X
# X_test = pd.DataFrame(X_test)
# y_test = pd.DataFrame(y_test)
# X_test.insert(2304, "class", y_test, allow_duplicates=False)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

X_train.insert(2304, "class", y_train, allow_duplicates=False)
#
#
# #change class attribute to nominal type
# # X_test["class"] = X_test["class"].map({0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5", 6: "s6", 7: "s7", 8: "s8", 9: "s9"})
X_train["class"] = X_train["class"].map({0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5", 6: "s6", 7: "s7", 8: "s8", 9: "s9"})
#
# #combine the 4000 instance with old test set
# # frame = [X_test, testSet]
# # result = pd.concat(frame)
#
# #output to cvs file
# # result.to_csv(r'resource/newData/xy_test9000_smpl.csv', index = False, header=True)
X_train.to_csv(r'resource/newData/xy_train9000_smpl.csv', index = False, header=True)


