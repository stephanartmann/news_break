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
gf.fetch(10)

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
cl.k_means(20)

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