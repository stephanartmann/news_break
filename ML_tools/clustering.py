#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:50:45 2020

@author: me
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

'''
Class to perform clustering.
'''
class Clusterer:
    '''
    Instances are initialized with a data frame 'article_vectors', where
    each row corresponds to an article, and each column to a variable/feature.
    '''
    def __init__ (self,article_vectors):
        self.article_vectors = article_vectors
        self.clusters = pd.Series(np.nan*article_vectors.index)
        self.clusters.index = article_vectors.index
        self.KMeans = None
        
    '''
    Perform k-means with 'n_clusters' many clusters.
    Result is stored in member 'clusters'. It is a pandas Series object
    with member labels, whose index corresponds to the original articles.
    '''
    def k_means (self,n_clusters=7):
        self.KMeans = KMeans(n_clusters=n_clusters)
        self.clusters = pd.Series(self.KMeans.fit_predict(self.article_vectors))
        self.clusters.index = self.article_vectors.index