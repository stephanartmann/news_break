#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 08:36:27 2021

@author: me
"""


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,make_pipeline
from umap import UMAP


'''
    Choose which columns of article data frame shall be used.
    Fit_transform of article data returns list of two data frames:

    First data frame corresponds to text data and has columns as in
    'text_columns'. 
    
    Second data frame corresponds to additional non-text data, such as dates,
    and contains columns as in 'non_text_columns'.
'''
class DataHandler(TransformerMixin, BaseEstimator):
    def __init__(self,text_columns=['Title','Article','Summary'],
                 nontext_columns=['Date'],source = None):
        self.text_columns = text_columns
        self.nontext_columns = nontext_columns
        self.source = source
        
    def fit(self,articles,y=None):
        self.X_text = articles[self.text_columns]
        self.X_nontext = articles[self.nontext_columns]
        print ('DH-Fit')
        return self
    
    def transform(self,articles,y=None):
        print ('DH-transform')
        return [self.X_text,self.X_nontext]
    
    def fit_transform(self,articles,y=None):
        self.fit(articles)
        return self.transform(articles)
    
'''
    Wrapper Class for Transformers that do support sklearn syntax,
    but cannot handle pass-through extra variables.
'''
class TransformerWrapper(TransformerMixin,BaseEstimator):
    
    def _initialize(self,wrapped_class,used_article_element=0,**kwargs):
        self.wrapped_class = wrapped_class
        self.used_article_element = used_article_element
        
        self.transformer = wrapped_class(**kwargs)
    
    def fit(self,X,y=None):
        self.transformer.fit(X[self.used_article_element])
        return self 
    
    def transform(self,X,y=None):
        return [self.transformer.transform(X[self.used_article_element]),X[1]]
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)