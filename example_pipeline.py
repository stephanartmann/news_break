#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:54:18 2021

@author: me
"""


from googlenews.gnews import GNews_fetcher
from ML_tools.dimension_reduction import dim_reducer
from ML_tools.clustering import Clusterer
from ML_tools.pipelines import DataHandler
from ML_tools.embedding import SpacyEmbedder,STembedder
from ML_tools.dimension_reduction import DistanceComputer,UMAP_wrapper
from umap import UMAP
from hdbscan import HDBSCAN
from ML_tools.clustering import HDBSCAN_wrapper


import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display

from sklearn import metrics
from sklearn.pipeline import make_pipeline

import pandas as pd

##################################
### Building workflow manually ###
##################################
'''
gn = GNews_fetcher('Resident evil 2', '24/01/2019', '12/20/2020',
                                 page_no = 2)
X0=gn.fit_transform(None)
dh = DataHandler()
X1=dh.fit_transform(X0)
st=STembedder()
X2=st.fit_transform(X1)
dc=DistanceComputer()
X3=dc.fit_transform(X2)
um=UMAP_wrapper(n_components=2)
X4=um.fit_transform(X3)
plotdf=pd.DataFrame(X4[0])
plotdf.plot.scatter(0,1)
X5=dc.fit_transform(X4)
hdb=HDBSCAN_wrapper()
hdb.fit_transform(X5)
'''

pipe=make_pipeline(GNews_fetcher('Resident evil 2', '24/01/2019', '12/20/2020',
                                 page_no = 20),DataHandler(),STembedder(),
                   DistanceComputer(),UMAP_wrapper(n_components=10),
                   DistanceComputer(),HDBSCAN_wrapper(),None,
                   memory='/home/me/Downloads/del')
res=pipe.fit_transform(1)
