#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:54:18 2021

@author: me
"""


from googlenews.gnews import *
from ML_tools.dimension_reduction import dim_reducer
from ML_tools.clustering import Clusterer
from ML_tools.pipelines import DataHandler
from ML_tools.embedding import SpacyEmbedder,STembedder
from umap import UMAP


import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display

from sklearn import metrics
from sklearn.pipeline import make_pipeline


pipe=make_pipeline(GNews_fetcher('Resident evil 2', '24/01/2019', '12/20/2020',
                                 page_no = 2),DataHandler(),STembedder(),UMAP(n_components=20),
                   memory='/home/me/Downloads/del')
res=pipe.fit_transform(None)
