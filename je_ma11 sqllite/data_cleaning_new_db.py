
# Data Wrangling and Cleaning
# Data Wrangling and Cleaning
# Importing libraries
import sqlite3
import pandas as pd
import numpy as np
# Creating sqlite connection.
cnx = sqlite3.connect('Cars_latest.db')
# Reading data in to dataframe
df = pd.read_sql_query("SELECT * FROM cars", cnx)

#Checking Null values

df.owners.fillna(value=np.nan, inplace=True)
df.owners = df.owners.replace(np.nan, '0 Owners')


# Removing string part from Owners and Depreciation 
df['owners'] = df['owners'].map(lambda x: x.lstrip('').rstrip('Owners'))
df['depreciation'] = df['depreciation'].map(lambda x: x.lstrip('$').rstrip('/yr'))
df['depreciation'] = df['depreciation'].str.replace(',','')
df.mileage =pd.to_numeric(df.mileage, errors='coerce').fillna(0).astype(int)
df['depreciation'] = df['depreciation'].map(lambda x: x.lstrip('Paper Value : $').rstrip(''))

# Converting Strings to integer columns (price	mileage	owners	depreciation)

df["price"] = pd.to_numeric(df["price"])
df["mileage"] = pd.to_numeric(df["mileage"])
df["owners"] = pd.to_numeric(df["owners"])
df["depreciation"] = pd.to_numeric(df["depreciation"])

# Splitting coe_left in to individual Registration and coe_left columns
df[['reg_date','years_left']] = df.coe_left.str.split("(",expand=True)
# Changing coe_left from string object to Date 
df['years_left'] = df['years_left'].map(lambda x: x.lstrip('').rstrip('mths left)'))
df['years_left'] = df['years_left'].map(lambda x: x.lstrip('').rstrip(' mths COE)'))

df.years_left.unique()
df[['coe_year','coe_months']] = df.years_left.str.split("yrs ",expand=True)

df["coe_year"] = df['coe_year'].str.replace('yr','')
#Replacing nan  values
df = df.fillna(0)
df["coe_year"] = pd.to_numeric(df["coe_year"])
df["coe_months"] = pd.to_numeric(df["coe_months"])
df['coe_months'] = df['coe_months'].div(12)

df.reg_date.unique()
df['years_left']= df[['coe_year','coe_months']].sum(axis=1)
df["reg_date"] = df['reg_date'].str.replace(' ','')
df['reg_date'] = pd.to_datetime(df['reg_date'])
df['reg_date'] = pd.to_datetime(df['reg_date']).dt.normalize()
df_cleaned = df[['id', 'brand', 'model', 'website', 'price', 'mileage', 'owners',
       'depreciation', 'coe_left','reg_date','years_left', 'link']]
df_cleaned.head()
#Creating a cursor object using the cursor() method
cursor = cnx.cursor()

#Doping EMPLOYEE table if already exists
cursor.execute("DROP TABLE cars")
print("Table dropped... ")
df_cleaned.to_sql(name='cars', con=cnx)
df = pd.read_sql_query("SELECT * FROM cars", cnx)
df.head()
