import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import tree
import warnings
warnings.simplefilter("ignore")

# FIRST PART
# CODE WILL NOT RUN. THIS IS JUST FOR REFERENCE

def main():
    # Reading csv file in to a dataframe
    df = pd.read_csv("ccDefaults.csv")
    #print(df.head())
    #print(df.shape)

    # print(df.isnull().sum())
    # drop any null values, not to interfere with complete data
    df.dropna(inplace=True)
    # No null values, no need for data wrangling for null values

    # Dropping id column 
    df.drop("ID", axis=1, inplace=True)
    print(df.head())

    # Getting correlation to get 4 features with greatest correlation with dpnm
    cor = df.corr()

    #Correlation with output variable
    cor_target = abs(cor["dpnm"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.21]
    print(relevant_features)

    # Creating new dataframe with correlated variables and target feature

    df_cor = df[ ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4','dpnm'] ]
    print(df_cor)
    descriptive_features = df_cor[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
    target_feature = df_cor['dpnm']
    # Standardizing the dataframe  
    df_cor[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']] = StandardScaler().fit_transform(df_cor[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']])
    # df_cor
    # Breaking into target and descriptive features
    descriptive_features = df_cor[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
    target_feature = df_cor[['dpnm']]

    # partition dataset into 70-30

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(descriptive_features, target_feature, test_size = 0.3,random_state = 2021)
    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=2021)
    # fit the model
    clf_en.fit(X_train, y_train)
    y_pred_en = clf_en.predict(X_test)
    # Accuracy of test set
    print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))
    # accuracy of train set
    y_pred_train_en = clf_en.predict(X_train)
    #y_pred_train_en
    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_en, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['1','0'])
    cmd.plot()
    # plotting tree
    plt.figure(figsize=(16,12))
    tree.plot_tree(clf_en.fit(X_train, y_train)) 


if __name__ == '__main__':
    main()
