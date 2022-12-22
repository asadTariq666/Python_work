import pandas as pd
pd.set_option('display.max_columns', None)
owners = pd.read_csv('owners.csv', low_memory = False)
companies = pd.read_csv('companies.csv')

companies_clean = companies.drop(columns=["Unnamed: 0", "GEO_ID", "NAICS2017","SEX","ETH_GROUP", "RACE_GROUP", "VET_GROUP", "EMPSZFI","FIRMPDEMP_F", "FIRMPDEMP_F", "RCPPDEMP_F", "EMP_F", "PAYANN_F", "FIRMPDEMP_S_F", "RCPPDEMP_S_F", "EMP_S_F", "PAYANN_S_F"], axis=1)


owners_clean = owners.drop(columns=["Unnamed: 0", "GEO_ID", "NAICS2017","OWNER_SEX","OWNER_ETH", "OWNER_RACE", "OWNER_VET", "YEAR", "QDESC", "OWNCHAR", "OWNPDEMP_F", "OWNPDEMP_PCT_F","OWNPDEMP_S_F", "OWNPDEMP_PCT_S_F", "state"], axis=1)

owners_clean.to_csv("owners_clean.csv")
companies_clean.to_csv("companies_clean.csv")
