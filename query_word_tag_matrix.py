# -*- coding: utf-8 -*-
import cPickle
import re
from collections import Counter

def find_word_count_on_all_tags(word_tag_dict, word):
    total_occurance = 0
    
    regex = re.compile(r"\s%s\s"%word) # \s is whitespace
    for key, value in word_tag_dict.iteritems():   # iter on both keys and values
        if word in key.split():
#                print key, value
            total_occurance = total_occurance + value
    print total_occurance
    return total_occurance

def create_word_cardinality_list(word_tag_dict):
    word_counter = Counter()
    
def find_tag_count_on_all_words(word_tag_dict, tag):
    total_occurance = 0
    
    for key, value in word_tag_dict.iteritems():   # iter on both keys and values
        if tag in key.split():
#                print key, value
            total_occurance = total_occurance + value
    print total_occurance
    return total_occurance

def create_sorted_word_tag_tuple_list(word_tag_dict):
    return sorted(word_tag_dict.items(), key=lambda x: x[1], reverse=True)

# Given the list of tags for each doc, creates a dictionary counting tags.
def count_tags_from_tag_file(tag_list):
    tag_counter = Counter()    
    for doc in tag_list:
        tag_counter.update(doc.split())
        print 'Number of tags in counter: '+str(len(tag_counter.keys()))
    
    f = open('Tag_Counts.data','wb')
    cPickle.dump(tag_counter,f,-1)

# It sorts the given dict in decreasing orders and return tuple list of words 
# and their cardinality number
def sort_given_word_tag_dict(word_tag_dict):
    return sorted(word_tag_dict.items(), key=lambda v: v[1], reverse=True)
