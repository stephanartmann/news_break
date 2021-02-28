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


import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display

from sklearn import metrics
from sklearn.pipeline import make_pipeline

reload_data = True

if (reload_data):
    gf = GNews_fetcher('Resident evil 2', '24/01/2019', '12/20/2020')
    page_no = 2
    gf.fetch(page_no)

pipe=make_pipeline(DataHandler(),STembedder(),None,memory='/home/me/Downloads/del')
res=pipe.fit_transform(gf.articles)
