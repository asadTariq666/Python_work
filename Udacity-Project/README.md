# Project - Wrangling and analyzing data from twitter archives
Part of Udacity's Data Analyst Nanodegree

## Project Overview
Gathered data from the archives of a twitter account called WeRateDogs which rates owned dogs in their tweets and adds a humorous comment with it. Gathered retweet and like counts for all the tweets from the twitter API using the access library tweepy and read them into a pandas dataframe in the Jupyter Notebook. Downloaded a tsv file about the tweets programmatically using the requests and BeautifulSoup libraries in python. Assessed quality and tidiness issues in the the gathered datasets and cleaned all of them using pandas functions. Analysed the cleaned tables and created visualizations using the matplotlib library.

## What would you need to install?
To work on the project, you will need to install either Python 3 or Anaconda.

## Project Files
twitter-archive-enhanced : Csv file containing data about the tweets like tweet text, dog ratings, dog name, dog type, etc. 

image-predictions : Tsv file containing all the dog breed predictions for all the tweets, computed using a neural network that predicted the breeds by using the dog images in the tweets. Other features include image url and breed_predicted which has two values True and False, True for when a breed was predicted and False if otherwise. 

tweet_json : A text file created which consists of each tweet's json object.

tweet_counts_clean : A file created from a cleaned dataset consisting of retweet and like counts for all the tweets. These counts were gathered using the twitter API and the python library tweepy.

image_predictions_clean : Cleaned version of the image-predictions file.

twitter_archive_master : A master dataset created by joining the cleaned versions of all the three files.

wrangle_act : Project workbook where all the wrangling, analyzing and visualizing was done.

wrangle_report : Project report discussing all the activities performed in the workbook.

act_report : Report communicating the insights drawn from the wrangled datasets with the help of visualizations.
