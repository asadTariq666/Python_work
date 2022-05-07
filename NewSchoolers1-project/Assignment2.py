# Assignment 2
# Personal Loan Prediction Using Trees
# Use the UniversalBank.csv dataset for this assignment

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.simplefilter("ignore")

def main():
    # Load the data from the file UniversalBank.csv. (2)
    df = pd.read_csv("UniversalBank.csv")
    print(df.head())
    #print(df.isnull().sum()) 
    # No null values, no need for data wrangling for null values

    # •	   What is the target variable? (2)
    print("Target variable is : Personal Loan")


    # •	Remove the attributes Row and Zip code. (3)
    df = df.drop(columns=['Row', 'ZIP Code'])
    print(df.head())
    ## All columns except Personal Loan are descriptive features
    descriptive_features = df[["Age",  "Experience"  ,"Income",  "Family",  "CCAvg",  "Education",  "Mortgage","Securities Account"  ,"CD Account",  "Online",  "CreditCard"]]
    target_feature = df[["Personal Loan"]]
    target_feature
    # partition dataset into 70-30
    # •	Partition the dataset:
    # 	•	random_state = 42 (1)
    # 	•	Partitions 70/30 (1)
    # 	•	Make sure to stratify! (1)


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(descriptive_features, target_feature, test_size = 0.3,random_state = 42,stratify=target_feature)
    #X_train.shape,y_train.shape,X_test.shape
    #How many of the cases in the training partition represented people who accepted offers of a personal loan? (3)

    count = y_train['Personal Loan'].sum()
    print('count of people in training set who accepted offers of a personal loan:', count)
    # Plot the classification tree Use entropy criterion. Max_depth = 5, random_state = 42. (4)

    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    # fit the model
    clf_en.fit(X_train, y_train)
    y_pred_en = clf_en.predict(X_train)
    plt.figure(figsize=(16,12))
    tree.plot_tree(clf_en.fit(X_train, y_train)) 


    # On the training partition, how many acceptors did the model classify as non-acceptors? (3)
    # On the training partition, how many non-acceptors did the model classify as acceptors? (3)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_train, y_pred_en)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Accepted','Rejected'])
    cmd.plot()

    print('acceptors classified as non-acceptors in training set: ',cm[0][1])
    print('Non-Acceptors classified as acceptors in training set: ',cm[1][0])
    # What was the accuracy on the training partition? (2)

    # Accuracy of Training set
    print('Model accuracy score with criterion entropy of Training set: {0:0.4f}'. format(accuracy_score(y_train, y_pred_en)))
    y_pred_en_test = clf_en.predict(X_test)
    # Accuracy of Testing set
    print('Model accuracy score with criterion entropy of Test set: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en_test)))

if __name__ == '__main__':
    main()
