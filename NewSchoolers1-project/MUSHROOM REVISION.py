import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
    

# easily read data
pd.set_option('display.width', None)
file_path = 'mushrooms.csv'
df = pd.read_csv(file_path)

x = df.iloc[:, 1:]
x_dummies = pd.get_dummies(x)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(x_dummies, y, test_size=0.30, random_state=42, stratify=y)

dct = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
# fit the model
dct.fit(X_train, y_train)
# SEE IF THERE IS DIFFERENCE X TRAIN
y_pred = dct.predict(X_test)

plot_confusion_matrix(dct, X_test, y_pred)  
plt.savefig('Confusion_Matrix.png')

# GOING VERBATIM HERE
# #5: What was the accuracy on the training partition? (100% accuracy)
y_pred2 = dct.predict(X_train)
acc_score_train = metrics.accuracy_score(y_train, y_pred2)
print(acc_score_train)

# What was the accuracy on the testing partition? (100% accuracy)
acc_score_test = metrics.accuracy_score(y_test, y_pred)
print(acc_score_test)
# Both scores indicate optimal performance

# Build and show the classification tree
names = x_dummies.columns
plt.figure(figsize=(15,10))
tree.plot_tree(dct, max_depth=6, feature_names=names, fontsize=10)
plt.savefig('Classification_Tree.png')


# #8 List the top three most important features in your decision tree for determining toxicity
#important_features = pd.Dataframe(zip(X_train.columns, dct.feature_importances_))
#print(important_features.sort_values(by=[1]))
# Based on this printout, I would say odor_n, stalk-root_c, and stalk-surface-below-ring_y
# are the most important features for determining toxicity

# #9 Classify the following mushroom
mushroom = {'cap-shape':'x', 'cap-surface':'s', 'cap-color':'n', 'bruises':'t',
            'odor':'y', 'gill-attachment':'f', 'gill-spacing':'c', 'gill-size':'n',
            'gill-color':'k', 'stalk-shape':'e', 'stalk-root':'e', 'stalk-surface-above-ring':'s',
            'stalk-surface-below-ring':'s', 'stalk-color-above-ring':'w', 'stalk-color-below-ring':'w',
            'veil-type':'p', 'veil-color':'w', 'ring-number':'o', 'ring-type':'p',
            'spore-print-color': 'r', 'population':'s', 'habitat':'u'}

# add new mushroom to our feature frame
x_new = x.append(mushroom)
x_new_dummies = pd.get_dummies(x_new)
object = np.array([x_new_dummies.iloc[-1, :]]).reshape(1, -1)
print('We would classify this new mushroom as', dct.predict(object))
