import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# first part
# SVM
def part_one():
    dataset = [[0, 0, 'A'], [1, 1, 'A'], [2, 3, 'B'], [2, 0, 'A'], [3, 4, 'B']]
    df_nums = pd.DataFrame(dataset, columns=['x1', 'x2', 'label'])
    print(df_nums)


    A = df_nums.loc[df_nums['label'] == 'A', :]
    B = df_nums.loc[df_nums['label'] == 'B', :]
    plt.scatter(A['x1'], A['x2'], color='red', label='red')
    plt.scatter(B['x1'], B['x2'], color='blue', label='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM example')

    X = df_nums[ ['x1', 'x2'] ]
    y = df_nums['label']

    model_svm = SVC(kernel='linear')
    model_svm.fit(X, y)

    weights = model_svm.coef_[0]
    bias = model_svm.intercept_
    sup_vecs = model_svm.support_vectors_
    slope = -weights[0]/weights[1]

    hyperplane_x = np.linspace(0, 4)
    hyperplane_y = slope*hyperplane_x -bias/weights[1]

    plt.plot(hyperplane_x, hyperplane_y, color='green', label='hyperplane')

    print('weights:', weights)
    print('bias:', bias)
    print('support vectors:', sup_vecs)
    print('slope:', slope)

    new_data = [[2, 1]]
    new_pred = model_svm.predict(new_data)
    print(new_data, 'prediction:', new_pred)
    plt.scatter(new_data[0][0], new_data[0][1], color='purple')


    plt.tight_layout()
    plt.grid(visible=True)
    plt.legend()
    plt.savefig('svm.png')


if __name__ == '__main__':
    part_one()




# second part
# KKN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    # import the wine quality dataset
    pd.set_option('display.width', None)
    file_path = 'winequality.csv'
    df_wines = pd.read_csv(file_path)
    # print(df_wines.head())

    # plot the count of wine quality
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(df_wines['Quality'])
    ax[0].set(xlabel='Quality', ylabel='Count', title='Histogram of Quality')

    # set the factors (x) and response (y) arrays
    y = df_wines['Quality']
    x = df_wines.drop(['Quality'], axis=1)
    # print(x.head())

    # normalize the factors (X)
    norm = Normalizer()
    X = pd.DataFrame(norm.fit_transform(x), columns=x.columns)
    # print(X.head())

    # partition the dataset
    # train | validation | test
    #  60   |     20     |  20
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=42, train_size=0.6, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, random_state=42, train_size=0.5, stratify=y_temp)

    # iterate on different ks (1 to 100), train, predict, gather accuracies
    ks = range(1, 101)
    scores = []
    for k in ks:
        # import and instantiate algorithm
        model_knn = KNeighborsClassifier(n_neighbors=k)
        model_knn.fit(X_train, y_train)
        # y_pred = model_knn.predict(X_valid)
        score = model_knn.score(X_valid, y_valid)
        # print('score:', score)
        scores.append(score)

    # plot accuracy as a function of k
    ax[1].plot(ks, scores)
    ax[1].set(xlabel='K neighbors', ylabel='Accuracy Score', title='Accuracy for various k\'s')


    fig.tight_layout()
    plt.savefig('wine_viz.png')


if __name__ == '__main__':
    main()



# part three
# decision trees
# look into how feature importances are calcuated!
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def main():
    pd.set_option('display.width', None)
    file_path = 'iris.csv'
    df_iris = pd.read_csv(file_path)
    # print(df_iris.head())

    corr_matrix = df_iris.corr()
    print('correlation matrix:\n', corr_matrix)

    y = df_iris['Species']
    X = df_iris.drop('Species', axis=1)
    # print(X.head())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

    model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model_dt.fit(X_train, y_train)

    y_pred = model_dt.predict(X_test)
    # print(y_test[:5])
    # print(y_pred[:5])

    acc_score = model_dt.score(X_test, y_test)
    # print('accuracy score:', acc_score)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # cm_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model_dt.classes_)
    # fig, ax1 = plt.subplots()
    # cm_disp.plot(ax=ax1)
    # plt.tight_layout()
    # plt.savefig('iris_cm.png')

    class_rep = classification_report(y_test, y_pred)
    # print(class_rep)

    plot_tree(model_dt, feature_names=X.columns, class_names=y.unique(), filled=True)

    plt.tight_layout()
    plt.savefig('iris_tree.png')

    fi = model_dt.feature_importances_
    print('features:', X.columns)
    print('feature importances:', fi)

    
