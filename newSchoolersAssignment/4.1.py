import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

# FIRST PART
# CODE WILL NOT RUN. THIS IS JUST FOR REFERENCE

def main():
    # Reading csv file in to a dataframe
    df = pd.read_csv("St[ores.csv")
    print(df.head())
    print(df.isnull().sum()) 
    # No null values, no need for data wrangling for null values

    # extracting descriptive features
    df_desc  = df.iloc[:,1:]
    print(df_desc)

    #Normalization
    norm = Normalizer()
    df_ft = pd.DataFrame(norm.fit_transform(df_desc), columns=df_desc.columns)
    print(df_ft)
    
    # Plotting inertia vs k
    ks = range(1, 10)
    inertias = []

    for k in ks:
        model_kmeans = KMeans(n_clusters=k, random_state=2021)
        model_kmeans.fit(df_ft)
        inertias.append(model_kmeans.inertia_)
    
    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(ks, inertias, marker='o')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('inertia')
    ax[0].set_title('inertia with respect to k')


    # optimal k is 3 as per the elbow method above.
    # Clustering with k=3

    model_kMeans = KMeans(n_clusters=3)

    model_kMeans.fit(df_ft)

    print(model_kMeans.labels_)

    # adding back to dataframe
    df['Cluster'] = model_kMeans.labels_

    print(df.head(5))
    # Histogram
    
    ax[1].hist(df['Cluster'])
    ax[1].set_xlabel('Cluster label')
    ax[1].set_ylabel('count of observations')
    ax[1].set_title('Cluster Histogram')

if __name__ == '__main__':
    main()
