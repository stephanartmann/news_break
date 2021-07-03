#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:01:33 2021

@author: me
"""

import sqlite3
import http.cookiejar 
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd

def get_cookies(cj, ff_cookies):
    con = sqlite3.connect(ff_cookies)
    cur = con.cursor()
    cur.execute("SELECT host, path, isSecure, expiry, name, value FROM moz_cookies")
    for item in cur.fetchall():
        c = http.cookiejar.Cookie(0, item[4], item[5],
            None, False,
            item[0], item[0].startswith('.'), item[0].startswith('.'),
            item[1], False,
            item[2],
            item[3], item[3]=="",
            None, None, {})
        print (c)
        cj.set_cookie(c)
        


#################################################
### Parse RSS-Feed to get articles and topics ###
#################################################

rss = 'https://news.google.com/rss?hl=de&gl=CH&ceid=CH:de'
feed = feedparser.parse(rss)

main_links = []
for entry in feed.entries:
    for link in BeautifulSoup(entry['summary']).findAll('a'):
        if (link.get('href').startswith('https://news.google.com/stories')):
            main_links.append(link.get('href'))


#####################################
### Download articles from topics ###
#####################################

cookie_path = '/home/me/.mozilla/firefox/1uv55f7w.python/cookies.sqlite'
cj = http.cookiejar.CookieJar()
get_cookies(cj,cookie_path)

links = []
i = 1
for URL in main_links:
    print ('Preprocessing Google Story',str(i),'of',str(len(main_links)))
    soup = BeautifulSoup(requests.get(URL,cookies=cj).content,'html.parser')
    
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    
    i = i+1
    
art = pd.Series(links).astype(str)
print(len(art[art.str.startswith('./articles')]))
