# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter("ignore")


def main():

    # Load the data from the file auto-mpg.csv.
    file_path = 'auto-mpg.csv'
    df = pd.read_csv(file_path)
    #print(df.head())

    # summarize the data set
    summarize = df.describe()
    #print(summarize)
    
    # what is the mean of mpg?
    mean = df['mpg'].mean()
    print('Mean of MPG is: ', mean)
    # what is the median value of mpg?
    median = df['mpg'].median()
    print('Median of MPG is: ', median)

    # which value is higher, mean or median? 
    print('Mean is slightly bigger than median')
    
    fig, ax = plt.subplots(3)
    
    
    # plot a histogram
    # to find skewness of attribute values
    ax[0].hist(df['mpg'])
    ax[0].set(title='Value ranges for MPG')
    print('\nValues for mpg are skewed to the right')
    print('This is in accordance with a mean larger than median')
    
#   ****NOTE****
#   I was having trouble with overlapping graphs
#   So I fixed the issue by putting questions 4-7 at the end of code
#   I understand this is bad practice--but the outputs are still in line!
#   I hope this is a nonissue!

    
    
    # STARTING WITH #8
    # No is the first in dataset, car_name is the last
    # can use iloc to remove
    df_desc = df.iloc[:,1:-1]
    #df.drop(labels='car_name', axis=1, inplace=True)
    #df.drop(labels='No', axis=1, inplace=True)

    # build a linear regression model with mpg as the target and displacement as the predictor 
    predictor = df_desc[['displacement']]
    target = df_desc[['mpg']]
    predictor.head(), target.head()

    # fit multiple linear regression lines to the data set
    regressor = LinearRegression()
    regressor.fit(predictor, target)

    # what is the value of the intercept β0?
    print('\nThe value of the intercept β0 is: ',regressor.intercept_[0])

    # what is the value of the coefficient β1 of the attribute displacement? (1)
    print('\nThe value of the coefficient β1 is: ',regressor.coef_[0][0])


    # what is the regression equation as per the model?
    print('\nThe regression equation as per the model is : y = 35.175 + (-0.06x)')

    # does the predicted value for mpg increase or decrease as the displacement increases?
    print('\nThe coefficient is negative')
    print('Which means the predicted value for MPG decreases as displacement increases')

    # given a car with a displacement value of 220, what would your model predict its mpg to be?
    pred = regressor.predict([[220]])
    print('\nGiven a car with a displacement value of 220, mpg would be: ', pred[0][0])

    # display a scatterplot of the actual mpg v displacement
    ax[1].scatter(predictor, target, color='blue', marker='.')
    # superimpose the linear regression line
    ax[1].plot(predictor, regressor.predict(predictor), color='red')
    ax[1].set(title='Displacement V. MPG', xlabel='Displacement', ylabel='MPG')
    ax[1].tight_layout()
    ax[1].savefig('Plot_1_8.png')

    #ax[2].plot(predictor, regressor.predict(predictor), color='red')
    sns.residplot(predictor, regressor.predict(predictor))
    
    # BACK TO QUESTIONS 4-7
    
    # plot the pairplot matrix of all the relevant numeric attributes
    # using df_desc from above
    sns.pairplot(df_desc)
    plt.savefig('Pair_Plot.png')
    
    # correlation matrix information
    #corr = df_desc.corr()
    #sns.heatmap(corr)
    #plt.show()
    
    # which two attributes seem to be most strongly linearly correlated?
    # NOTE: I am saying anything strongly correlated is closest to 1 OR -1
    print('Features that are most strongly correlated: Horsepower and MPG')
    # NOTE: runner ups include weight and horsepower, and displacement and cylinders
    
    # NOTE: I am saying anything weakly correlated is closest to 0, either positive or negative
    # which two attributes seem to be most weakly correlated
    print('\nFeatures that are most weakly correlated: Weight and Model year')
    # other runner ups include weight/acceleration, and mpg/acceleration
    
    
    # produce a scatterplot of the two attributes mpg and displacement with displacement on the x axis and mpg on the y axis. (2)
    plt.scatter(df_desc['displacement'], df_desc['mpg'])
    plt.title('Displacement v MPG')
    plt.xlabel('Displacement')
    plt.ylabel('Miles per Gallon')
    # NOTE: for smooth code, commented this out
    # so no overlapping graphs
    #plt.savefig('MPG_Displacement.png')


if __name__ == '__main__':
    main()
