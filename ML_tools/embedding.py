#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 08:42:15 2021

@author: me
"""

import spacy
from sentence_transformers import SentenceTransformer

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


'''
Perform dimension reduction on 'columns' of data frame 'articles' 
that contains the text data.
'''
class SpacyEmbedder(TransformerMixin,BaseEstimator):
    '''
    Initialized with 'articles' data frame, language (throws an error if not one of ['en']),
    and text columns that shall be considered.
    '''
    def __init__ (self,lang='en',normalize=True):
        self.mapped_text_vectors = None
        self.spacy_nlp = None
        if (lang == 'en'):
            self.spacy_lib = 'en_core_web_lg'
        else:
            raise ValueError("'lang' must be one of ['en']")
        
        self.normalize=normalize
        self.lang = lang
    
    '''
    Use spacy to vectorize and store result in (article x dimension) data frame,
    accessible as member 'mapped_text_vectors'.
    WARNING: If normalization is applied, the vectors are divided by the 
    *length* of the text.
    '''
    def _spacy_vectorization(self,normalize=False):
        if (self.spacy_nlp is None):
            self.spacy_nlp = spacy.load(self.spacy_lib)
        articles = self.articles[0]
        text_vector_df = pd.DataFrame()
        for column in articles.columns:
            mapped = {}
            dim = 0
            for text in articles[column]:
                mapped[dim] = self.spacy_nlp(text).vector
                if (normalize):
                    mapped[dim] = mapped[dim]/len(text)
                dim = dim + 1
            text_vector_df = pd.concat([text_vector_df,pd.DataFrame(mapped,
                            index=column + pd.Series([str(i) for i in range(0,len(mapped[0]))]))])
        text_vector_df.columns = articles.index
        print ('Vectorization called')
        print (self.articles[0].shape)
        self.mapped_text_vectors = text_vector_df.transpose()
        
    def fit(self,articles,y=None):
        self.articles = articles
        print ('Fit called')
        print (self.articles[0].shape)
        return self
    
    def transform(self,articles,y=None):
        print ('Transform called')
        print (self.articles[0].shape)
        self._spacy_vectorization(normalize=self.normalize)
        return [self.mapped_text_vectors,self.articles[1]]
    
    def fit_transform(self,articles,y=None):
        self.fit(articles)
        return self.transform(articles)




'''
Use SentenceTransformer to embed texts
'''
class STembedder(TransformerMixin,BaseEstimator):
    name2model = {
        'multi':'distiluse-base-multilingual-cased-v2',
        'en':'stsb-roberta-large'}
    
    def __init__(self,language='en'):
        self.language = language
        self.model_name = self.name2model[language]
        self.embedder = SentenceTransformer(self.model_name)
        
    def fit(self,articles,y=None):
        self.articles = articles
        return self
    
    def transform(self,articles,y=None):
        text_data  = articles
        text_vector_df = pd.DataFrame()
        for column in text_data.columns:
            embeddings = pd.DataFrame(self.embedder.encode(text_data[column]))
            embeddings.columns = column+pd.Series(embeddings.columns).astype(str)
            text_vector_df = pd.concat([text_vector_df,embeddings],axis=1)
        self.mapped_text_vectors = text_vector_df
        return self.mapped_text_vectors
    
    def fit_transform(self,articles,y=None):
        self.fit(articles)
        return self.transform(articles)
        