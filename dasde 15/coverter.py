# Importing Libraries
import pandas as pd 
import numpy as np
import warnings
warnings.simplefilter("ignore")

## Load Dataset
df = pd.read_excel('red1.xlsx')
df.head()
## dataframe with all rows data
df_all =df
# keeping data when condition = y
df = df.loc[df['Condition'] == 'y']
df.head(10)

# Getting dateID data in an array
x = np.array(df_all['dateId'])
r = np.array(df_all['dateId2'])
s =np.array(df_all['dateId3'])



# Getting a list of counts of every unique number in the dateId Column
y = df_all['dateId'].value_counts()
# Getting the keys of the list of counts
keys = y.keys()
# setting a counter with values of keys
counters = y.values

# Declaring new arrays to populate values in the redial column
redial =[]
redial2 =[0]*len(r)
redial3 =[0]*len(s)

# redial 1 column
for i in range(len(x)):
    for j  in range(len(counters)):
        if x[i]==keys[j]:
            counters[j]=counters[j]-1
            redial.append(counters[j])

# updating the counter 
y = df_all['dateId'].value_counts()
keys = y.keys()
counter2 = y.values

# redial 2 column
for i in range(len(r)):
    for j  in range(len(counter2)):
        if r[i]==keys[j]:
            redial2[i]=counter2[j]


# redial 3 column
for i in range(len(s)):
    for j  in range(len(counter2)):
        if s[i]==keys[j]:
            redial3[i]=counter2[j]



# Assigning '-' to rows where condition = 'n'
for i in range(len(df_all)):
    if df_all['Condition'][i] == 'n':
        redial[i]= '-'
        redial2[i]= '-'
        redial2[i]= '-'
        
# Assigning values to dataframe
df_all.redial1 = redial
df_all['redial2'] = redial2
df_all['redial3'] = redial3


#saving in to excel and csv file
df_all.to_csv('Updated File.csv')
df_all.to_excel('Updated File.xlsx')

