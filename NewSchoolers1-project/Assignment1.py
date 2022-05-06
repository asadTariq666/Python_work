import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.simplefilter("ignore")


def main():
    # Load the data from the file winequality.csv. (2)
    df = pd.read_csv("Winequality.csv")
    print(df.head())
    print(df.isnull().sum()) 
    # No null values, no need for data wrangling for null values
    # extracting descriptive features
    df_desc  = df.iloc[:,:-1]
    print(df_desc.head())
    # extracting target feature
    target_feature  = df.iloc[:,-1:]
    print(target_feature.head())
    # Standardize all variables other than Quality. (2)
    norm = Normalizer()
    descriptive_features = pd.DataFrame(norm.fit_transform(df_desc), columns=df_desc.columns)
    descriptive_features.head()
    #Partition the dataset:
	    #•	random_state = 42 (1)
	    #•	Partitions 60/20/20 (1)
	    #•	Make sure to stratify! (1)
    x, x_test, y, y_test = train_test_split(descriptive_features,target_feature,test_size=0.2,train_size=0.8,random_state=42,stratify=target_feature)
    x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.25,train_size =0.75,random_state=42,stratify=y)
    #print(x_train.shape,x_test.shape,x_cv.shape)
    #Iterate on K ranging from 1 to 30.
    #	•	Build a KNN classification model to predict Quality based on all the remaining numeric variables. (2)
    #	•	Plot the accuracy for both the Training and Validation datasets. (4)


    # Training set
    ks = list(np.arange(1,30+1))
    score = []
    # looping through 1 - 30
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        score.append(metrics.accuracy_score(y_test, y_pred))
    print(score)
    # plot the relationship between K and testing accuracy
    plt.figure(1)
    plt.plot(ks, score)
    plt.xlabel('Value of K ')
    plt.ylabel('Accuracy of Testing')

    # Validation set
    ks2 = ks
    score2 = []
    # looping through 1 - 30
    for k2 in ks2:
        knn2 = KNeighborsClassifier(n_neighbors=k2)
        knn2.fit(x_cv, y_cv)
        y_pred2 = knn2.predict(x_test)
        score2.append(metrics.accuracy_score(y_test, y_pred2))
    print('printing score 2 ',score2)
    # plot the relationship between K and testing accuracy
    
    plt.figure(2)
    plt.plot(ks2, score2)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy Testing')

    # Which value of k produced the best accuracy in the Training and Validation data sets? (2)
    # Getting max value of score in training set
    max_value = max(score)
    max_index = score.index(max_value) +1
    print("Best value of k for Training set:",max_index)

    # Getting max value of score in validation set
    max_value2 = max(score2)
    max_index2 = score2.index(max_value2) +1
    print("Best value of k for Validation set:",max_index2)
    # Generate predictions for the test partition with the chosen value of k. 
    # Plot the confusion matrix of the actual vs predicted wine quality. (4)

    knn3 = KNeighborsClassifier(n_neighbors=1)
    knn3.fit(x_train, y_train)
    y_pred3 = knn3.predict(x_test)
    score3 = metrics.accuracy_score(y_test, y_pred3)
    #score3

    # df.Quality.unique()

    #Generate predictions for the test partition with the chosen value of k. 
    #Plot the confusion matrix of the actual vs predicted wine quality. (4)
    cm = confusion_matrix(y_test, y_pred3)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['3','4','5','6','7','8'])
    print(cmd.plot())
    acc_score = accuracy_score(y_test, y_pred3)
    print('accuracy of model on the test dataset is : ', acc_score *100 )
    y_test['Predicted Quality'] = y_pred3 
    print(y_test.head())


if __name__ == '__main__':
    main()
