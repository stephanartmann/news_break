#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:25:05 2021

@author: me
"""

import gradio as gr

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
import numpy as np



from wordcloud import STOPWORDS
wc = WordCloud(background_color='white', width=1000, height=400, stopwords=STOPWORDS)

def main_call(keywords,starting_date,end_date,no_of_google_pages):
    '''
        
    
        Parameters
        ----------
        keyword : TYPE
            DESCRIPTION.
        starting_date : TYPE
            DESCRIPTION.
        end_date : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
    print('Started function')
    pipe=make_pipeline(GNews_fetcher(keywords, starting_date, end_date,
                                 page_no = no_of_google_pages),DataHandler(),STembedder(),
                   DistanceComputer(),UMAP_wrapper(n_components=10),
                   DistanceComputer(),HDBSCAN_wrapper(),None,
                   memory='/home/me/Downloads/del')
    res=pipe.fit_transform(1)
    articles = pipe.named_steps['gnews_fetcher'].articles.Article
    
    white_img = np.zeros([100,100,3],dtype=np.uint8)
    white_img.fill(255) 
        
    imagelist=[white_img,white_img,white_img,white_img,white_img]
    for i in range(min(5,max(res.label))):
        group = res[res.label !=-1].label.value_counts().index[i]
        print("Wordcloud for {}".format(group))
        print('Cluster size:',sum(res.label == group))
        imagelist[i]=wc.generate(" ".join(t for t in articles[res.label == group])).to_image()
    return imagelist[0],imagelist[1],imagelist[2],imagelist[3],imagelist[4]

'''
keywords = 'resident evil 2'
starting_date = '01/01/2018'
end_date = '01/01/2019'
'''

iface = gr.Interface(
  fn=main_call, 
  inputs=["text","text","text","number"],
  outputs=["image", "image", "image", "image", "image"])
iface.launch()

