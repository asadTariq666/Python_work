import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np
# Importing Linear regression from sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter("ignore")


def main():

    # Reading csv file in to a dataframe
    games = pd.read_csv("vgsales.csv")
    print(games.head())
    print(games.shape)

    #sorting by year
    games_sorted = games.sort_values(by = ['Year'])

    #We look at the popular genres in each region
    genre_NA = games_sorted.groupby('Genre').NA_Sales.sum()
    genre_NA_sorted = genre_NA.sort_values(ascending = False)
    genre_EU = games_sorted.groupby('Genre').EU_Sales.sum()
    genre_EU_sorted = genre_EU.sort_values(ascending = False)
    genre_JP = games_sorted.groupby('Genre').JP_Sales.sum()
    genre_JP_sorted = genre_JP.sort_values(ascending = False)
    genre_other = games_sorted.groupby('Genre').Other_Sales.sum()
    genre_other_sorted = genre_other.sort_values(ascending = False)

    print(genre_NA_sorted[[0]],genre_EU_sorted[[0]], genre_JP_sorted[[0]],genre_other_sorted[[0]])
    
    #getting games of top genre from each region
    games_NA = games.loc[(games['Genre'] == 'Action') ]
    games_EU = games.loc[(games['Genre'] == 'Action') ]
    games_JP = games.loc[(games['Genre'] == 'Role-Playing') ]
    games_other = games.loc[(games['Genre'] == 'Action') ]

    # NA Region - Genre Action
    df_NA_Action = games_NA.groupby(['Year'], as_index = False)
    df_NA_Action_mean = df_NA_Action[['NA_Sales']].aggregate(np.mean)
    df_NA_Action_mean.head()

    descriptive_feature = df_NA_Action_mean[['Year']]
    target_feature = df_NA_Action_mean[['NA_Sales']]
    regressor = LinearRegression(normalize=True)
    regressor.fit(descriptive_feature, target_feature)

    viz = plt
    viz.scatter(descriptive_feature, target_feature, color='red')
    viz.plot(descriptive_feature, regressor.predict(descriptive_feature), color='blue')
    viz.title('Action in NA')
    viz.xlabel('Year ')
    viz.ylabel('Sales')
    viz.show()

    # EU Region - Genre Action
    df_EU_Action = games_EU.groupby(['Year'],as_index = False)
    df_EU_Action_mean = df_EU_Action[['EU_Sales']].aggregate(np.mean)
    df_EU_Action_mean.head()

    descriptive_feature = df_EU_Action_mean[['Year']]
    target_feature = df_EU_Action_mean[['EU_Sales']]
    regressor = LinearRegression(normalize=True)
    regressor.fit(descriptive_feature, target_feature)
    viz = plt
    viz.scatter(descriptive_feature, target_feature, color='red')
    viz.plot(descriptive_feature, regressor.predict(descriptive_feature), color='blue')
    viz.title('Action in EU')
    viz.xlabel('Year ')
    viz.ylabel('Sales')
    viz.show()


    # JP Region - Genre Role Playing
    df_JP_Action = games_JP.groupby(['Year'],as_index = False)
    df_JP_Action_mean = df_JP_Action[['JP_Sales']].aggregate(np.mean)
    df_JP_Action_mean.head()

    descriptive_feature = df_JP_Action_mean[['Year']]
    target_feature = df_JP_Action_mean[['JP_Sales']]
    regressor = LinearRegression(normalize=True)
    regressor.fit(descriptive_feature, target_feature)
    viz = plt
    viz.scatter(descriptive_feature, target_feature, color='red')
    viz.plot(descriptive_feature, regressor.predict(descriptive_feature), color='blue')
    viz.title('Role-Playing in JP')
    viz.xlabel('Year ')
    viz.ylabel('Sales')
    viz.show()

    # Other Region - Genre Action

    df_other_Action = games_other.groupby(['Year'],as_index = False)
    df_other_Action_mean = df_other_Action[['Other_Sales']].aggregate(np.mean)
    df_other_Action_mean.head()

    descriptive_feature = df_other_Action_mean[['Year']]
    target_feature = df_other_Action_mean[['Other_Sales']]
    regressor = LinearRegression(normalize=True)
    regressor.fit(descriptive_feature, target_feature)
    viz = plt
    viz.scatter(descriptive_feature, target_feature, color='red')
    viz.plot(descriptive_feature, regressor.predict(descriptive_feature), color='blue')
    viz.title('Action in Other Countries')
    viz.xlabel('Year ')
    viz.ylabel('Sales')
    viz.show()   

    
if __name__ == '__main__':
    main()
