import pandas as pd
import requests 

get = "GEO_ID,NAME,NAICS2017,NAICS2017_LABEL,OWNER_SEX,OWNER_SEX_LABEL,OWNER_ETH,OWNER_ETH_LABEL,OWNER_RACE,OWNER_RACE_LABEL,OWNER_VET,OWNER_VET_LABEL,QDESC,QDESC_LABEL,OWNCHAR,OWNCHAR_LABEL,YEAR,OWNPDEMP,OWNPDEMP_F,OWNPDEMP_PCT,OWNPDEMP_PCT_F,OWNPDEMP_S,OWNPDEMP_S_F,OWNPDEMP_PCT_S,OWNPDEMP_PCT_S_F"

key = 'fd8c137d32e065e499223dba1534a3c5e7a52b91'

geography = 'state'

url = (f"https://api.census.gov/data/2018/abscbo?get={get}" +

        f"&for={geography}:" +

        f"*&key={key}")



r = requests.get(url)

data = r.text



owners = pd.read_json(data)



owners.columns = owners.iloc[0]

owners = owners[1:]
pd.set_option('display.max_columns', None)


get1 = "GEO_ID,NAME,NAICS2017,NAICS2017_LABEL,SEX,SEX_LABEL,ETH_GROUP,ETH_GROUP_LABEL,RACE_GROUP,RACE_GROUP_LABEL,VET_GROUP,VET_GROUP_LABEL,RCPSZFI_LABEL,EMPSZFI,EMPSZFI_LABEL,YEAR,FIRMPDEMP,FIRMPDEMP_F,RCPPDEMP,RCPPDEMP_F,EMP,EMP_F,PAYANN,PAYANN_F,FIRMPDEMP_S,FIRMPDEMP_S_F,RCPPDEMP_S,RCPPDEMP_S_F,EMP_S,EMP_S_F,PAYANN_S,PAYANN_S_F"

key = 'fd8c137d32e065e499223dba1534a3c5e7a52b91'

geography1 = 'state'

url1 = (f"https://api.census.gov/data/2018/abscs?get={get1}" +

        f"&for={geography1}:" +

        f"*&key={key}")



r1 = requests.get(url1)

data1 = r1.text



companies = pd.read_json(data1)



companies.columns = companies.iloc[0]

companies = companies[1:]
pd.set_option('display.max_columns', None)


owners.to_csv("owners.csv")
companies.to_csv("companies.csv")




