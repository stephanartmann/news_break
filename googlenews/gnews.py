#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:52:36 2020

@author: Stephan Artmann
"""

from twitter.tweets import *
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import nltk

from sklearn.base import BaseEstimator, TransformerMixin



user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent


'''
Class for Google News articles.
'''

class GNews_fetcher(TransformerMixin,BaseEstimator):
    '''
    When creating instance, specify a string of keywords, a start and an end date 
    (string of format MM/DD/YYYY). Only articles matching the keywords and with
    publication date between the start and the end date will be considered.
    '''
    def __init__(self,keyword_string,start_date,end_date,page_no=10,data_set_name=None,lang='en'):
        self.articles = None
        self.start_date = start_date
        self.end_date = end_date
        self.raw_articles = None
        self.keyword_string = keyword_string
        self.page_no = page_no
        self.data_set_name = data_set_name
        self.lang = lang
        
        self.tweeterers = {}
        self.retweeterers = {}
        
        self.article2tweeter = {}
    
    '''
    Fetch Articles from Google News and store their link, title, description, 
    publication date, and full text in the data frame 'articles'.
    
    page_no: Number of pages from Google News to be fetched, defaults to 10.
    
    ToDo: fix datetime of articles.
    '''
    def fetch(self,page_no = 10,lang='en'):
        googlenews=GoogleNews(start=self.start_date,end=self.end_date,lang=lang)
        googlenews.search(self.keyword_string)
        result=googlenews.result()
        df=pd.DataFrame(result)
        count = 1
        for i in range(2,page_no+1):
            print ('Fetched page:',count)
            count = count + 1
            googlenews.getpage(i)
            result=googlenews.result()
            df=pd.concat([df,pd.DataFrame(result)])
        print ('Fetched page:',count)
        df = df.drop_duplicates()
        df = df.reset_index().drop(columns=['index'])
        self.raw_articles = df
        list=[]
        for ind in df.index:
            print('Downloading article ',ind+1,' of ',df.shape[0])
            try:
                dict={}
                article = Article(df['link'][ind],config=config)
                article.download()
                article.parse()
                article.nlp()
                dict['Date']=df['date'][ind]
                dict['Media']=df['media'][ind]
                dict['Title']=article.title
                dict['Article']=article.text
                dict['Summary']=article.summary
                dict['Link'] = df['link'][ind]
                dict['Datetime'] = df['datetime'][ind]
                list.append(dict)
            except Exception as e:
                print ('Failed for: ',df.link[ind])
                print (e)
        news_df=pd.DataFrame(list)  
        self.articles = news_df

    def get_twitter(self):
        tf = TweetFetcher()
        self.tweeterers = {}
        self.article2tweeter = {}
        for i,link in enumerate(self.articles.Link):
            print('Article ',i,' of ',self.articles.shape[0])
            tf.get_tweets(link)
            article_tweeterers = {}
            for tweeter in tf.user2tweet_id.keys():
                if (tweeter in self.tweeterers.keys()):
                    self.tweeterers[tweeter] = self.tweeterers[tweeter]+1
                else:
                    self.tweeterers[tweeter] = 1
                if (tweeter in article_tweeterers.keys()):
                    article_tweeterers[tweeter] = article_tweeterers[tweeter]+1
                else:
                    article_tweeterers[tweeter] = 1
            self.article2tweeter[link] = article_tweeterers

    
    def fit(self,X=None,y=None):
        self.fetch(self.page_no,lang=self.lang)
        return self
    
    def transform(self,X=None,y=None):
        return self.articles
    
    def fit_transform(self,X=None,y=None):
        self.fit()
        return self.transform()
    

    