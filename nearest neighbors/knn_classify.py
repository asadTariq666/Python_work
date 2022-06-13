
import pandas as pd
import numpy as np
from math import dist
import os
import sys
from pip import main
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
clear = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

def normalize_data(data_set):
    
    training_v_mean_df = pd.DataFrame()
    training_mean_df = data_set[:-1].iloc[:, :-1].mean().to_frame().T
    last_col = data_set[16].to_list()

    for index, columns in (data_set.iloc[:,:-1].iteritems()):
        clear()
        print("CALCULATING V-MEAN VALUES.....")
        print("DIMENSION :", index+1,"\n")
        temp_series = pd.Series([])
        for i in tqdm(range(len(columns))):
            temp_series = temp_series.append(columns[i]-training_mean_df[index], ignore_index = True)
        training_v_mean_df[index] = temp_series

    squared_v_mean_df = np.square(training_v_mean_df)
    df_mean = squared_v_mean_df.mean().to_frame().T
    std_df = np.sqrt(df_mean)
    preprocessed_training_df =  training_v_mean_df.div(std_df.iloc[0],axis = 'columns')
    preprocessed_training_df.insert(len(preprocessed_training_df.columns), len(preprocessed_training_df.columns), last_col)

    return preprocessed_training_df
   
class KNN():

    def __init__(self, training, K):
        self.training = training
        self.K = K

    def prediction_classification(self, test_row):
        self.test_row = test_row

        distances = list()
        for i in self.training.values():
            train_row = list()
            for j in i.values():
                train_row.append(j)
            distance = dist(self.test_row[:-1], train_row[:-1])
            distances.append((distance, train_row[-1]))
        distances = sorted(distances)
        self.neighbors = [distances[i][1] for i in range(self.K)]
        prediction = max(set(self.neighbors), key=self.neighbors.count)
        self.prediction = prediction
        if hasattr(self, "prediction"):
            return self.prediction
    
    def accuracy_finder(self):
        
        accuracy_dict = {i:self.neighbors.count(i) for i in self.neighbors}
        temp = []
        for label, count in accuracy_dict.items():
            if count == max(accuracy_dict.values()):
                temp.append(label)
        
        if self.prediction == self.test_row[16]:
            if len(temp) > 1:
                accuracy = 1/len(temp)
            else:
                accuracy = 1
        else:
            accuracy = 0
        
        return accuracy

        
def main():
    current_directory = os.getcwd()
    
    training_df = pd.read_fwf(current_directory+"//"+sys.argv[1], header= None)
    testing_df = pd.read_fwf(current_directory+"//"+sys.argv[2], header = None)
 
    Normalized_training_df = normalize_data(training_df)
    Normalized_testing_df = normalize_data(testing_df)
    Accuracy_sum = 0.0

    training_dict = Normalized_training_df.T.to_dict()
    testing_dict = Normalized_testing_df.T.to_dict()

    KNN_model = KNN(training_dict, int(sys.argv[3]))    

    clear()
    for index, value in (testing_dict.items()):
        test_row = list()
        for j in value.values():
            test_row.append(j)
        prediction = KNN_model.prediction_classification(test_row)
        Accuracy = KNN_model.accuracy_finder()
        Accuracy_sum += Accuracy*100
        print("ID=", index, "   \tPredicted=", int(prediction), "   \tTrue=", int(test_row[16]))
    final_accuracy = Accuracy_sum/len(testing_dict.values())

    print("\n\nClassification Accuracy=", float("{0:.4f}".format(final_accuracy)))

if __name__ == "__main__":
    main()
