# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys

assert sys.version_info >= (3, 5)
import sklearn
# assert sklearn.__version__ >= "0.20"  REMOVING THIS BECAUSE ASSERT SKLEARN IN VMWARE IS OLD

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay

# Common imports
import numpy as np
import os


def print_hi(attributes):
    # Use a breakpoint in the code line below to debug your script.

    if attributes == 2:
        a = 2
        depth = 2
    elif attributes == 4:
        a = 0
        depth = 3

    iris = load_iris()
    X = iris.data[:, a:]  # petal length and width   # <--------- modify here --------
    y = iris.target

    # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf.fit(X_train, y_train)

    rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    rnd_clf.fit(X_train, y_train)

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)

    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    models = [("DT", tree_clf), ("RF", rnd_clf), ("NB", gnb_clf), ("SVM", svm_clf)]

    unsorted_scores = [(name, cross_val_score(model, X_train, y_train, cv=10).mean()) for name, model in models]
    scores = sorted(unsorted_scores, key=lambda x: x[1])
    print(scores)
    high_Val = (max(scores))

    if high_Val[0] == 'SVM':
        clf_pred = svm_clf.predict(X_test);
    elif high_Val[0] == 'DT':
        clf_pred = tree_clf.predict(X_test)
    elif high_Val[0] == 'NB':
        clf_pred = gnb_clf.predict(X_test)
    elif high_Val[0] == 'RF':
        clf_pred = rnd_clf.predict(X_test)

    print(accuracy_score(y_test, clf_pred))

    cm = confusion_matrix(y_test, clf_pred)
    print("The confusion Matrix is:")
    print(cm)
    #cm_display = ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()
    #print(cm_display)

    print(classification_report(y_test, clf_pred, target_names=iris.target_names))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("1st output for classifier with 2 attributes")
    attributes = 2
    print_hi(attributes)
    print("2nd output for classifier with 4 attributes")
    attributes = 4
    print_hi(attributes)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
