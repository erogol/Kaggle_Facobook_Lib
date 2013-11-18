# -*- coding: utf-8 -*-

from BeautifulSoup import BeautifulSoup as bs 
import scipy as sc 
import numpy as np 
import pandas as pn 
from HTMLParser import HTMLParser
from joblib import Parallel, delayed, Memory
from multiprocessing import Pool
from itertools import repeat
from feature_extraction import *



def clear_data_parallel(file_path):
    DATA = pn.read_csv(file_path)
    
#    mem = Memory(cachedir='tmp')
#    memgist = mem.cache(foo)

#    DATA.Body = Parallel(n_jobs=30, verbose=11)(delayed(memgist)(doc,counter)for counter,doc in enumerate(DATA.Body))
    poll = Pool(processes = 30)    
    
    result = poll.map(foo,zip(DATA.Body, range(0,DATA.Body.shape[0])))
    DATA['Body'] = result
    DATA.to_csv(file_path+'_no_code.csv',index=False)
    print('Cleaning is finished!!!')
    
def foo((doc,counter)):
    print(counter)
    soup = bs(doc)
    all_tags = soup.findAll('code')
    if(len(all_tags)>0):
        for  tag_in in all_tags:
            tag_in.replaceWith('')
    return strip_tags(soup.getText())
    
'''
    Remove HTML tags from given text
'''
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

if __name__ == '__main__':
    clear_data_parallel('DATA/Train.csv')
#    extract_bow_features('DATA/Train_no_code.csv')