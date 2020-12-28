#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:01:37 2020

@author: me
"""

from googlenews.gnews import *
from ML_tools.dimension_reduction import dim_reducer
from ML_tools.clustering import Clusterer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display

from sklearn import metrics

gf = GNews_fetcher('Resident evil 2', '24/01/2019', '12/20/2020')
page_no = 2
gf.fetch(page_no)

########################
### Get Twitter data ###
########################

gf.get_twitter()

numbers = pd.DataFrame({'tweeter':gf.tweeterers.keys(),
                        'freq':gf.tweeterers.values()})

#####################################
### Word embedding and clustering ###
#####################################
dr = dim_reducer(gf.articles)
dr.spacy_vectorization()

cl = Clusterer(dr.mapped_text_vectors)
cl.k_means(min([7,len(gf.articles.Link)]))

print(cl.KMeans.inertia_)
print(metrics.silhouette_score(dr.mapped_text_vectors, cl.clusters, metric='euclidean'))
print(metrics.silhouette_score(dr.mapped_text_vectors, cl.clusters))



from wordcloud import STOPWORDS
wc = WordCloud(background_color='white', width=1000, height=400, stopwords=STOPWORDS)

for group in cl.clusters.unique():
    print("Wordcloud for {}".format(group))
    print('Cluster size:',sum(cl.clusters == group))
    for ind in gf.articles.index[cl.clusters == group]:
        print(gf.articles.loc[ind].Link,gf.articles.loc[ind].Date)
    display(wc.generate(" ".join(t for t in gf.articles.Article[cl.clusters == group])).to_image())
    
    
    
#####################################################
### Testing clustering algorithm on Covid dataset ###
#####################################################

df = pd.read_excel('example_data/padiweb_covid19.xlsx',engine='openpyxl')
dr = dim_reducer(df,columns=['text'])
dr.spacy_vectorization()

cl = Clusterer(dr.mapped_text_vectors)
cl.k_means(len(df.rss_feed_content.unique()))

df['clusters'] = cl.clusters
res=df.groupby(['clusters','rss_feed_content']).aggregate({'one':'sum'}).reset_index().pivot(index='clusters',columns='rss_feed_content',values='one').fillna(0)
res.to_excel('example_data/covid_clustering.xlsx')


########################################
### Test algorithm on Kaggle dataset ###
########################################

pd.read_csv('https://www.kaggle.com/kotartemiy/topic-labeled-news-dataset/download/7xUkJPKE0eRKOqPD2HOl%2Fversions%2FbKrBDOCLPC6aLobiUWMu%2Ffiles%2Flabelled_newscatcher_dataset.csv')
