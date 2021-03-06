#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:20:44 2020

@author: me
"""

import spacy
import pandas as pd
from umap import UMAP


'''
Perform dimension reduction on 'columns' of data frame 'articles' 
that contains the text data.
'''
class dim_reducer:
    '''
    Initialized with 'articles' data frame, language (throws an error if not one of ['en']),
    and text columns that shall be considered.
    '''
    def __init__ (self,articles,lang='en',columns=['Title','Article','Summary']):
        self.articles = articles.copy()
        self.columns = columns
        self.mapped_text_vectors = None
        self.spacy_nlp = None
        if (lang == 'en'):
            self.spacy_lib = 'en_core_web_lg'
        else:
            raise ValueError("'lang' must be one of ['en']")
    
    '''
    Use spacy to vectorize and store result in (article x dimension) data frame,
    accessible as member 'mapped_text_vectors'.
    WARNING: If normalization is applied, the vectors are divided by the 
    *length* of the text.
    '''
    def spacy_vectorization(self,normalize=True):
        if (self.spacy_nlp is None):
            self.spacy_nlp = spacy.load(self.spacy_lib)
        text_vector_df = pd.DataFrame()
        for column in self.columns:
            mapped = {}
            dim = 0
            for text in self.articles[column]:
                mapped[dim] = self.spacy_nlp(text).vector
                if (normalize):
                    mapped[dim] = mapped[dim]/len(text)
                dim = dim + 1
            text_vector_df = pd.concat([text_vector_df,pd.DataFrame(mapped,
                            index=column + pd.Series([str(i) for i in range(0,len(mapped[0]))]))])
        text_vector_df.columns = self.articles.index
        self.mapped_text_vectors = text_vector_df.transpose()
            
