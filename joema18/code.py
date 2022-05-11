# Q3 Use Pandas and get average rating.

import pandas as pd  # importing pandas module
# Load the data from the IMDB-Movie-Data.csv.
df = pd.read_csv("/Users/asadtariq/Downloads/Python_work/Python_work/joema18/IMDB-Movie-Data.csv")
#print(df.head(2)) # viewing if the dataset is loaded or not
type(df['Rating']) # Checking type of Rating column
rating= df['Rating'].str.rstrip(" scores") # Extracting only numeric rating and discarding string part i.e. "8.1 score to 8.1"
print(rating) #printing ratings of all movies
print("Average of Ratings of all movies is: ",pd.to_numeric(rating).mean()) # Printing average of all ratings