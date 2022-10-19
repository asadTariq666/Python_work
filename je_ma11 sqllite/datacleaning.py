
# Data Wrangling and Cleaning
# Importing libraries
import sqlite3
import pandas as pd
# Creating sqlite connection.
cnx = sqlite3.connect('Carsupdate.db')
# Reading data in to dataframe
df = pd.read_sql_query("SELECT * FROM cars", cnx)

#Checking Null values
#print(df.isnull().sum())
#df.head()
# Removing string part from Owners and Depreciation 
df['owners'] = df['owners'].map(lambda x: x.lstrip('').rstrip('Owners'))
df['depreciation'] = df['depreciation'].map(lambda x: x.lstrip('$').rstrip('/yr'))
df['depreciation'] = df['depreciation'].str.replace(',','')
df['depreciation'] = df['depreciation'].map(lambda x: x.lstrip('Paper Value : $').rstrip(''))
#df.head()

# Converting Strings to integer columns (price	mileage	owners	depreciation)

df["price"] = pd.to_numeric(df["price"])
df["mileage"] = pd.to_numeric(df["mileage"])
df["owners"] = pd.to_numeric(df["owners"])
df["depreciation"] = pd.to_numeric(df["depreciation"])
# df.head()

# Splitting coe_left in to individual Registration and coe_left columns
df[['reg_date','coe_left']] = df.coe_left.str.split("(",expand=True)

# Changing coe_left from string object to Date 
df['coe_left'] = df['coe_left'].map(lambda x: x.lstrip('').rstrip('mths left)'))
# df.head()
df[['coe_year','coe_months']] = df.coe_left.str.split("yrs ",expand=True)

df["coe_year"] = df['coe_year'].str.replace('yr','')
df["coe_months"].unique()
#Replacing nan  values
df = df.fillna(0)
df["coe_year"] = pd.to_numeric(df["coe_year"])
df["coe_months"] = pd.to_numeric(df["coe_months"])
df['coe_months'] = df['coe_months'].div(12)
# df.head()
df['coe_left']= df[['coe_year','coe_months']].sum(axis=1)
df["reg_date"] = df['reg_date'].str.replace(' ','')
df['reg_date'] = pd.to_datetime(df['reg_date'], format='%d%b%y')
# df.head()

# Compiling the cleaned data
df_cleaned = df[['id', 'brand', 'model', 'website', 'price', 'mileage', 'owners',
       'depreciation', 'reg_date', 'coe_left', 'link']]
df_cleaned.head()
#Creating a cursor object using the cursor() method
cursor = cnx.cursor()

#Replacing Dirty car table with clean data
cursor.execute("DROP TABLE cars")

df_cleaned.to_sql(name='cars', con=cnx)
df = pd.read_sql_query("SELECT * FROM cars", cnx)
df.head()