import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def main():
    pd.set_option('display.width', None)
    file_path = "banking.csv"
    df_banking = pd.read_csv(file_path)
    # print(df_banking.info())
    # print(df_banking.isnull().sum())

    y = df_banking['y']
    # print(df_banking.head())


    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(y)
    ax[0, 0].set(title='result distribution', xlabel='result', ylabel='count')
    ax[1, 0].hist(df_banking['job'], orientation='horizontal')
    ax[1, 0].set(title='job distribution', xlabel='count', ylabel='job')
    ax[0, 1].hist(df_banking['marital'])
    ax[0, 1].set(title='marital distribution', ylabel='count', xlabel='marital status')
    fig.suptitle('Distributions')
    fig.tight_layout()

    plt.savefig('banking.png')

    print(df_banking.info())

    x = df_banking[ ['job', 'marital', 'default', 'housing', 'loan', 'poutcome'] ]
    print(x.info())

    X = pd.get_dummies(x)
    print(X.info())

    X = X.drop(X.columns[ [11, 15, 17, 20, 23] ], axis=1)
    print(X.info())

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    model_logreg = LogisticRegression(max_iter=100)

    model_logreg.fit(X_train, y_train)

    y_pred = model_logreg.predict(X_test)

    accuracy_score = model_logreg.score(X_test, y_test)

    print('accuracy score:', accuracy_score)



    cm = confusion_matrix(y_test, y_pred, labels=model_logreg.classes_)
    print(cm)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_logreg.classes_)
    fig, ax1 = plt.subplots() #figsize=(10,10))
    cm_disp.plot(ax=ax1)
    plt.savefig('banking_cm.png')


    class_rep = classification_report(y_test, y_pred)
    print(class_rep)

if __name__ == '__main__':
    main()
