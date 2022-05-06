

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression


def rms_titanic():
    
    #pd.set_option('display.width', None)
    # set up file path for titanic data
    file_path = ('titanicTrain.csv')
    # create dataframe to read titanic data
    dataframe = pd.read_csv(file_path)

    df = dataframe[ ['Survived', 'Sex', 'Pclass', 'Age'] ]
    #print(df.info())

    # drop any null values, not to interfere with complete data
    df.dropna(inplace=True)
    
    # use four variables--survived, sex, pclass, age--have Histograms
    fig, ax = plt.subplots(2, 2)
    # plots first histogram
    ax[0, 0].hist(df['Survived'])
    # sets x and y labels for current histogram
    ax[0, 0].set(xlabel='Survived', ylabel='Count')
    # follows trend from above
    ax[1, 0].hist(df['Sex'])
    ax[1, 0].set(xlabel='Sex', ylabel='Count')
    ax[0, 1].hist(df['Pclass'])
    ax[0, 1].set(xlabel='Pclass', ylabel='Count')
    ax[1, 1].hist(df['Age'])
    ax[1, 1].set(xlabel='Age', ylabel='Count')
    # figure is titled
    fig.suptitle('Titanic Data: Histograms of Input Variables')
    fig.tight_layout()
    # histogram #1 complete
    plt.savefig('preview.png')


    x_var = df[['Survived', 'Sex', 'Pclass', 'Age']]
    #print(x.info())
    # sex is the only variable that is categorical
    #sex = df[['Sex']]
    # transform into dummy variable
    x = pd.get_dummies(x_var)
    print(df)





    
    



if __name__ == '__main__':
    rms_titanic()
