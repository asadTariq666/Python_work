import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

# FIRST PART
# CODE WILL NOT RUN. THIS IS JUST FOR REFERENCE

def main():
    pd.set_option('display.width', None)
    file_path = 'winequality.csv'
    df_wines = pd.read_csv(file_path)
    # print(df_wines.head())
    # print(df_wines.isnull().sum())

    norm = Normalizer()

    # df_t = pd.DataFrame(norm.transform(df_wines), columns=df_wines.columns)
    df_ft = pd.DataFrame(norm.fit_transform(df_wines), columns=df_wines.columns)


    ks = range(1, 11)
    inertias = []

    for k in ks:
        model_kmeans = KMeans(n_clusters=k)
        model_kmeans.fit(df_ft)
        inertias.append(model_kmeans.inertia_)

    plt.plot(ks, inertias, marker='o')
    plt.savefig('wine_kmeans.png')

if __name__ == '__main__':
    main()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

#SECOND PART

def main():
    pd.set_option('display.width', None)
    file_path = 'winequality.csv'
    df_wines = pd.read_csv(file_path)
    print(df_wines.head())

    # print(df_wines.isnull().sum())

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(df_wines['Quality'])
    ax[0].set(title='Quality Histogram')

    df_wines_grouped = df_wines.groupby('Quality').mean()

    # print(df_wines_grouped)

    df_wines_norm = (df_wines-df_wines.min())/(df_wines.max() - df_wines.min())

    # print(df_wines_norm.head())

    model_ac = AgglomerativeClustering(n_clusters=6)

    model_ac.fit(df_wines_norm)

    # print(model_ac.labels_)

    ax[1].hist(model_ac.labels_)
    ax[1].set(title='Clusters Histogram')

    df_wines_norm['Cluster'] = model_ac.labels_

    # print(df_wines_norm.groupby('Cluster').mean())

    # print(df_wines.columns[0])
    df_wines.drop(df_wine.columns[0],axis=1,inplace=True)
    df_wines.drop('Fixed acidity',axis=1,inplace=True)
    df_wines.drop(,axis=1,inplace=True)


    plt.savefig('wine.png')

if __name__ == '__main__':
    main()

