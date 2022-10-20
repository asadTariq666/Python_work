## My Number incase there are any issues. +4917686525207

import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
# Creating sqlite connection.
cnx = sqlite3.connect('/Users/asadtariq/Downloads/Python_work/Python_work/je_ma11 sqllite/Carsupdate.db')
# Reading data in to dataframe
df = pd.read_sql_query("SELECT * FROM cars", cnx)

#changing reg_date to datetime type
df.reg_date=pd.to_datetime(df.reg_date)
now = pd.to_datetime('now')
now
df['age']= (now.year - df['reg_date'].dt.year)


# countplots function
def countplot(feature,df):
    sns.set(rc={'figure.figsize':(20,5)})
    sns.countplot(x=feature, data=(df)).set(title='Count of cars')
    plt.show()

# Bar Plot Function
def barplots(a,b,df):
    df.groupby(a)[b].mean().plot.bar()
    plt.title('Average '+b+ ' of each '+a)
    plt.legend(loc="upper left")
    plt.show()

# Pie Chart Function
def pieplots(a,feature):
    values=a.value_counts(dropna=True)
    plt.pie(a.value_counts(), autopct= lambda x: '{:.0f}'.format(x*values.sum()/100), startangle=90)
    plt.legend(a.value_counts().index)  
    plt.title('Pie Chart of ' + feature)
    plt.show()

# Line Plot Function
def lineplots(a,b,df):
    df.groupby(a)[b].mean().plot(kind='line', linestyle='--', marker='o', color='b',)
    plt.title('Average ' + b +' of car with respect to ' + a)
    plt.legend()
    plt.show()

# Density Plots
def densityplot(feature,df):
    sns.set(rc={'figure.figsize':(20,5)})
    sns.kdeplot(x=feature, data=(df)).set(title='Density of cars')
    plt.show()

def scatterplots():
    # Scatter Plots
    sns.scatterplot(data=df, y="brand", x="price").set(title='Scatter Plot of cars Price')
    plt.show()
    sns.scatterplot(data=df, y="brand", x="depreciation").set(title='Scatter Plot of cars Deprecation')
    plt.show()

# def relplots():
    # sns.relplot(data=df, x="price", y="brand", row="owners", height = 8).set(title='Comparison of Prices of Brands by Number of owners')
    # plt.show()

def proportionplots():
    # Proportion Plots
    sns.set(rc={'figure.figsize':(10,5)})
    sns.ecdfplot(data=df, y="owners").set(title='Proportion Division by numer of Owners')
    plt.show()
    sns.set(rc={'figure.figsize':(10,5)})
    sns.ecdfplot(data=df, y="brand").set(title='Proportion Division by Brands')
    plt.show()
    sns.set(rc={'figure.figsize':(10,5)})
    sns.ecdfplot(data=df, y="age").set(title='Proportion Division by Age of Car')
    plt.show()

def catplots():
    # Cat Plots
    sns.catplot(data=df, x="owners", y="brand", kind="box").set(title='Boxplot of Brands by number of owners')
    plt.show()
    sns.catplot(data=df, x="age", y="brand", kind="box").set(title='Boxplot of Brands by Age of car')
    plt.show()

def stripplots():
    # Strip Plots
    sns.stripplot(data=df, y="brand", x="price").set(title='Strip plot of Car Brands and Prices')
    plt.show()
    sns.stripplot(data=df, y="brand", x="depreciation").set(title='Strip plot of Car Brands and depreciation')
    plt.show()
    sns.stripplot(data=df, x="age", y="depreciation").set(title='Strip plot of Car Age and depreciation')
    plt.show()

def violinplots():
    sns.violinplot(data=df, x="age", y="depreciation").set(title='Violon plot of Car Age and depreciation')
    plt.show()
    sns.violinplot(data=df, x="owners", y="price").set(title='violin plot of Owners and Price')
    plt.show()
    sns.violinplot(data=df, x="owners", y="depreciation").set(title='violin plot of Owners and depreciation')
    plt.show()

def relationalplots():
    sns.lmplot(data=df, y="price", x="depreciation").set(title='Relation between Price and depreciation')
    plt.show()
    sns.lmplot(data=df, x="age", y="depreciation").set(title='Relation between Age and depreciation')
    plt.show()
    sns.lmplot(data=df, x="mileage", y="depreciation").set(title='Relation between Mileage and depreciation')
    plt.show()
    sns.lmplot(data=df, x="mileage", y="age").set(title='Relation between Age of cars and depreciation')
    plt.show()
    sns.lmplot(data=df, y="mileage", x="owners").set(title='Relation between Age of Cars and Owners')
    plt.show()

def heatmapplots():
    heatmap1_data = pd.pivot_table(df, values='mileage', index=['brand'], columns='age')
    sns.heatmap(heatmap1_data).set(title='Heatmap of each Brand with respect to age and Price')
    plt.show()




# Count Plots
print('Plotting Counts for each Brand: ', countplot(df.brand,df))
print('Plotting Counts for each Age: ', countplot(df.age,df))
print('Plotting Count of cars for each Number of owner: ', countplot(df.owners,df))

# Bar Plots
print('Plotting Barplot of brand and mileage: ', barplots('brand','mileage',df))
print('Plotting Barplot of brand and Prices: ', barplots('brand','price',df))
print('Plotting Barplot of brand and Owners: ', barplots('brand','owners',df))
print('Plotting Barplot of brand and Age of cars: ', barplots('brand','age',df))
print('Plotting Barplot of owners and Age of cars: ', barplots('owners','age',df))

#Pie Charts
print('Plotting Pie Chart of Car division by number of owners : ', pieplots(df.owners,'owners'))
print('Plotting Pie Chart of Car division by age of car : ', pieplots(df.age,'age'))
print('Plotting Pie Chart of Car division by brands of car : ', pieplots(df.brand,'brand'))

# Line Plots
print('Plotting Line Plots of between age and price of car : ', lineplots('age','price',df))
print('Plotting Line Plots of between age and Mileage of car : ', lineplots('age','mileage',df))
print('Plotting Line Plots of between age and depreciation of car : ', lineplots('age','depreciation',df))
print('Plotting Line Plots of between age and coe_left of car : ', lineplots('age','coe_left',df))
print('Plotting Line Plots of between brand and depreciation of car : ', lineplots('brand','depreciation',df))
print('Plotting Line Plots of between brand and coe_left of car : ', lineplots('brand','coe_left',df))

# Density Plots
print('Plotting Density of cars by number of owners: ', densityplot(df.owners,df))
print('Plotting Density of cars by  age: ', densityplot(df.age,df))
print('Plotting Density of cars by  price: ', densityplot(df.price,df))
print('Plotting Density of cars by mileage: ', densityplot(df.mileage,df))
print('Plotting Density of cars by depreciation: ', densityplot(df.depreciation,df))

# scatter plots
scatterplots()

#proportion plots
proportionplots()

# Cat Plots
catplots()

# Strip Plots
stripplots()

# Violon Plots
violinplots()

# Relational Plots
relationalplots()

# heatmap plots
heatmapplots()