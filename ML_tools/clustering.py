#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:50:45 2020

@author: me
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from hdbscan import HDBSCAN

from sklearn.base import BaseEstimator, TransformerMixin

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
        
        
class HDBSCAN_wrapper(TransformerMixin,BaseEstimator):
    def __init__(self,metric='precomputed',min_cluster_size=5,min_samples=None,
                 cluster_selection_epsilon=0.0,cluster_selection_method='eom'):
        self.metric=metric
        self.min_cluster_size=min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.clusterer=HDBSCAN(metric=metric,min_cluster_size=min_cluster_size,
                               min_samples=min_samples,
                               cluster_selection_epsilon=cluster_selection_epsilon,
                               cluster_selection_method = cluster_selection_method)
    def fit(self,X,y=None):
        self.daten = X
        self.clusterer.fit(X[0].astype('double'))
        return self
    def transform(self,X,y=None):
        print('Called HDBSCAN')
        self.Erg = pd.DataFrame({
            'label':self.clusterer.labels_,
            'prob':self.clusterer.probabilities_})
        return self.Erg
        
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
    