#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:54:47 2020

@author: Stephan Artmann
"""

import configparser
import snscrape.modules.twitter as sntwitter
import tweepy


# Read Twitter API-credentials from configuratino file
cp = configparser.RawConfigParser()   
configFilePath = r'twitter/twitter_config.txt'
cp.read(configFilePath)


'''
Class representing tweets.
'''
class Tweet:
    ID = 0;
    user = '';
    text = '';
    created_at = '';
    
    # Retweets of this tweet
    retweets = [];
    
    # Tweet that this Tweet is a Retweet of
    is_retweet_of = None;
    
    def __init__(self,sns_tweet=None):
        self.ID = 0;
        self.user = '';
        self.text = 0;
        self.created_at = 0;
        
        if (sns_tweet is not None):
            self.read_snscrape(sns_tweet)
        
    '''
    Generate tweet object from snscrape tweet object.
    '''
    def read_snscrape(self,sns_tweet,read_text=False,read_dates=False):
        self.ID = sns_tweet.id
        self.user = sns_tweet.user.username

    '''
    Generate tweet object from tweepy tweet object.
    '''
    def read_tweepy(self,tweepy_tweet):
        self.user = tweepy_tweet.user.screen_name
    
    
    '''
    Get retweets of this tweet.
    '''
    def get_retweets(self,tweepy_api):
        self.retweets = []
        retweets_list = tweepy_api.retweets(self.ID) 
        for retweet in retweets_list:
            rtw = Tweet()
            rtw.read_tweepy(retweet)
            self.retweets.append(rtw)
    


'''
Class to fetch tweets.
'''
class TweetFetcher:
    keyword_string = ''
    
    tweets = []
    tweet_ids = []
    user2tweet_id = {}
    
    retweet_ids = []
    retweet_users = []
    
    # Credentials for Official Twitter API. Are read from config file.
    credentials = {}

    
    
    def __init__(self):
        self.keyword_string = ''
        self.tweet_ids = []
        self.tweets = []
        
        self.user2tweet_id = {}
        
        self.retweet_ids = []
        self.retweet_users = []
        

        credential_conf_strings = ['consumer_key','consumer_secret','access_token',
                                   'access_token_secret']
        for cred in credential_conf_strings:
            self.credentials[cred] = cp.get('twitter-credentials',cred)
            
        
            
    
    '''
    Provided a string of keywords, retrieve all Tweet-IDs containing this string 
    and store them in self.tweet_ids.
    
    Additionally, the mapping of username to tweet-ID is stored in 
    the dicionary user2tweet_id.
    '''
    def get_tweets (self,keyword_string):
        self.tweet_ids = []
        self.tweets = []
        self.user2tweet_id = {}
        for tweet in sntwitter.TwitterSearchScraper(keyword_string).get_items():
            self.tweet_ids.append(tweet.id)
            self.user2tweet_id[tweet.user.username] = tweet.id
            tw = Tweet()
            tw.read_snscrape(tweet)
            self.tweets.append(tw)
         
    '''
    For all tweets in self.tweet_ids, fetch people who retweeted.
    '''
    def get_retweets (self):
        cred = self.credentials
        
        # authorization of consumer key and consumer secret 
        auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret']) 
          
        # set access to user's access key and access secret  
        auth.set_access_token(cred['access_token'], cred['access_token_secret']) 
          
        # calling the api  
        api = tweepy.API(auth) 
        
        self.retweet_ids = []
        self.user2retweet_id = {}
        
        count = 0
        for tweet in self.tweets:
            # getting the retweeters 
            tweet.get_retweets(api)
            print(count)
            count = count + 1
            for retweet in tweet.retweets:
                self.retweet_ids.append(retweet.ID)
                if (retweet.user not in self.retweet_users):
                    self.retweet_users.append(retweet.user)
            
            
  
