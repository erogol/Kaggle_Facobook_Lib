'''
Given the list of documents these functions are used for removing stop words
html entities, tokenizing, vocabulary creation (with most frequent words from 
given corpus) and BoW representation of the documents. All codes require NLTK 
library and STOPWORD corpus of it.
'''
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
import numpy as np
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
from collections import Counter
import re, htmlentitydefs
import string
import os
import glob
from multiprocessing import Pool, Array, Process, Queue, Manager
from joblib import Parallel, delayed
import pickle
import cPickle
import pandas as pn
import time 


"""
    PARALLELIZATION WITH PROCESSES
    Works on divided chunks of the original data and beeter for very large 
    scales contrained by the memoryc
    
"""
#####################################################
# FILE SYSTEM METHODS for CHUNKING or MERGING 
#####################################################


# Divide given csv file into given number of chunks and save the given path
def divide_data_into_chunks(file_name, output_path, num_chunks):
    DATA = pn.read_csv(file_name)
    num_ins = DATA.shape[0]
    chunk_size = np.floor(num_ins / num_chunks)
    residual = num_ins - chunk_size*num_chunks
    total_data = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in range(num_chunks):
        print 'Chunk ' + str(i+1) + ' of ' + str(num_chunks)
        print 'Start index '+ str(i*chunk_size)
        DATA_chunk = []
        if i == (num_chunks-1):
            DATA_chunk = DATA.iloc[i*chunk_size:((i+1)*chunk_size)+residual,]
        else:        
            DATA_chunk = DATA.iloc[i*chunk_size:(i+1)*chunk_size,]
        file_path = os.path.join(output_path,str(i)+'.csv')
        total_data = total_data+ len(DATA_chunk)
        print total_data
        DATA_chunk.to_csv(file_path,index=False)
    
    if total_data == num_ins:
        print 'Finished with GOOD!!'

def get_chunk_paths(chunk_path):
    path_list = glob.glob(chunk_path+'/*.csv')
    return path_list

def all_files_in(folder_path):
    path_list = glob.glob(folder_path+'/*')
    return path_list

def merge_word_tag_matrix_files(root_folder_path):
   files_path_list = all_files_in(root_folder_path)
   master_counter = Counter()
   for file_path in files_path_list:
       with open(file_path,'r') as f:
           temp_counter = cPickle.load(f)
           master_counter = temp_counter+master_counter
           print 'Number of items in matrix : ' + str(len(master_counter.keys()))
   cPickle.dump(master_counter,open('body_word_tag_matrix.data','w'))
   
   
################################################
###### PREPROCESSING METHODS - Parallelization via process creation
################################################
 # It creates chunks of instances in to separate files with tokenized and preprocesses word lists
def master_preprocessing_call(function ,path_chunks=None,processes=5, output_path_chunks=None, column='Body'):
    if path_chunks ==  None:
        raise Exception('Path should be defined')
    
    if not os.path.exists(output_path_chunks):
        os.makedirs(output_path_chunks)
        
    path_list = get_chunk_paths(path_chunks)
    num_files = len(path_list)
    process_list = range(processes)
    file_counter = 0
    write_counter = 0
    docs_list = range(processes)
    docs_queue = Queue()
    ids_list = [None]*processes
    file_names = [None]*processes
    while_flag = True
    while while_flag:   
        for process_no in range(processes):
            if file_counter == num_files:
                print('Processes all finished!!!')
                while_flag = False
                break;
#            print('File '+ str(file_counter)+' is queued----->')
            next_path = path_list[file_counter]
            file_names[process_no] = os.path.split(next_path)[1]
            out_file_path = output_path_chunks+'/'+file_names[process_no]
            docs_list[process_no] = list(np.array(pn.read_csv(next_path)[column]))
#            ids_list[process_no] = list(np.array(pn.read_csv(next_path)['Id']))
            process_list[process_no] = Process(target=function, args=(docs_list[process_no], process_no, out_file_path))
            file_counter = file_counter + 1
        
        for process in process_list:
            process.start()
        
        for process_no in range(processes):
#            temp_frame = pn.DataFrame({'Id':ids_list[process_no], 'Body':docs_list[process_no]})
#            temp_frame = temp_frame[['Id','Body']]
#            temp_frame.to_csv(output_path_chunks+'/'+file_names[process_no],index=False)
#            cPickle.dump(docs_queue.get(),open(output_path_chunks+'/'+file_names[process_no],'w'))
            write_counter = write_counter+1 
        
        for process in process_list:
            process.join()
        print('Number of files written : '+str(write_counter))
            
            
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# DUMMY CHILD PROCESS CALLS        
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            
def dummy_perform_tokenizing(docs, process_no, file_path):
    print str(process_no)+' started --- Tokenization!!!'
    tokenized_docs = []
    for (counter,doc) in enumerate(docs):     
        if doc != None:
#            print(type(doc))
            if type(doc) == float:
               doc = str(doc)
            docs[counter] = word_tokenize(doc)
        else:
            print('Doc '+str(counter+1)+' has None value!!!');
            docs[counter] = ' '
    cPickle.dump(docs,open(file_path,'w'))
    print str(process_no)+' finished!!!'
    
def dummy_remove_punctuation(docs, process_no, file_path):
    print str(process_no)+' started --- Punctuation Removal!!!'
    regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
    for counter,doc in enumerate(docs):
        new_doc = []
        for token in doc: 
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_doc.append(new_token.lower())
        docs[counter] = new_doc
    cPickle.dump(docs,open(file_path,'w'))
    print str(process_no)+' finished!!!'
    
def dummy_remove_stopwords(docs, process_no, file_path):
    print str(process_no)+' started --- Stopword Removal!!!'
#    new_docs = [None]*len(docs) 
    for counter,doc in enumerate(docs):
            new_term_vector = []
            for word in doc:
                if not word in stopwords.words('english'):
                    new_term_vector.append(word)
            docs[counter] = new_term_vector   
    cPickle.dump(docs,open(file_path,'w'))
    print str(process_no)+' finished!!!'

# Perform tokenization + stopword removal successively 
# and save the results for each file chunk as cPickle dump
def dummy_all_preprocessing(docs, process_no, file_path):
    new_docs = [None]*len(docs)
#    print str(process_no)+' started --- All Removal!!!'
#    regex = re.compile('[%s]' % re.escape(string.punctuation)) 
    for (counter,doc) in enumerate(docs):     
        if doc != None:
#            print(type(doc))
            if type(doc) == float:
               doc = str(doc)
            doc = word_tokenize(doc)
            new_token_vector = []
            for token in doc:
                if not token in stopwords.words('english'):
                    new_token_vector.append(token)
            new_docs[counter] = new_token_vector     
        else:
            print('Doc '+str(counter+1)+' has None value!!!');
            new_docs[counter] = ' '
    cPickle.dump(new_docs,open(file_path,'w'))
#    print str(process_no)+' finished!!!'


################################################
###### VOCABULARY GENERATION METHODS - Parellalization via process creation
################################################

def master_vocab_generation(function ,path_chunks=None ,output_path_data=None, processes=5, merge_processes = 5):
    if path_chunks ==  None:
            raise Exception('Path should be defined')
        
    path_list       = get_chunk_paths(path_chunks)
    num_files       = len(path_list)
    docs_list       = [None]*processes
    global_counter  = Counter()
    
    process_list        = [None]*processes
    merge_process_list  = [None]*merge_processes
    
    file_counter            = 0
    counter_queue           = Queue() 
    merged_counter_queue    = Queue()
    
    merge_process_list = [Process(target=dummy_merge_counter, args=(counter_queue, merged_counter_queue, output_path_data, counter, len(merge_process_list)))for counter,process in enumerate(merge_process_list)]
    
    for process in merge_process_list:    
        process.start()    
        
    while_flag = True
    while while_flag:
        process_list = []
        for process_no in range(processes):
            
            if file_counter == num_files:
                    while_flag = False
            
            next_path = path_list[file_counter]
#            docs_list[process_no] = list(np.array(pn.read_csv(next_path)['Body']))
#            docs_list[process_no] = cPickle.load(open(next_path,'r'))
            process_list.append(Process(target=function, args=(next_path, process_no, counter_queue)))
            file_counter = file_counter+1
        
        for process in process_list:
            process.start()
    
#        start = time.time()
#        tmp = Counter()
#        for process_no in range(processes):
#            tmp = tmp + counter_queue.get()
#        global_counter = global_counter + tmp
#        stop = time.time()
        if while_flag == False:
            for process in merge_process_list:
                counter_queue.put(-1) # stop signal
            for process in merge_process_list:
                process.join()
#        if while_flag == False:
#            print("FINISHED!!!")
#            f = open('Word_Counts.data','wb')
#            cPickle.dumps(global_counter, f, -1)
#            return global_counter
        print 'Number of files '+str(file_counter)
        
        
def dummy_merge_counter(counter_queue, merge_counter_queue, output_path_data, process_no, merge_processes_num):
    master_counter = Counter()
    while True:
        item = counter_queue.get()
#        print('Merging')
        if item == -1:
           cPickle.dump(master_counter, open(str(process_no) + output_path_data,'w'))
           merge_counter_queue.put(master_counter)
#           master_counter_queue.put(master_counter) 
           if process_no == 0:
               master_counter = Counter()
               print('Final MERGING!!!')
               for i in range(merge_counter_queue):
                   master_counter = master_counter + merge_counter_queue.get()
                   f = open('all_merged_'+ output_path_data,'wb')
                   cPickle.dumps(master_counter, f, -1)
                   print("FINISHED!!!")
           return
        master_counter = master_counter + item
        print('Process '+str(process_no)+' Number of items in matrix ' + str(len(master_counter.keys())))
    
def dummy_token_count(chunk_file, process_no, qu):
    # print str(process_no)+' started --- Token counting!!!'
    tf = Counter()
    f = open(chunk_file,'r')
    docs = cPickle.load(f)
    f.close()
    if docs == None        :
        raise Exception('Doc list should be given !!!')  
    for counter,doc in enumerate(docs):
#        print str(counter)
        try:
            for token in doc:
                tf[token] += 1
#            tf.update(doc)
        except:
            continue
    qu.put(tf)
#    print str(process_no)+' finished!!!'
    
    
    
    
"""
    PARALLELIZATION VIA MAPPING 
    Causes memory bloating for some large data sets
"""

################################################
###### PREPROCESSING METHODS - Emberisingly Parallelization via Mapping
################################################

# Step 1: perform tokenization on the raw document
def perform_tokenizing(raw_docs=None, parallel = True, threads = 5): 
    if raw_docs == None:
        raise Exception('Parameter error')
    
    
    print('Tokenization ...')    
    tokenized_docs = []
    
    if parallel:
        pool = Pool(processes=threads)
        tokenized_docs = pool.map(word_tokenize,map(str,raw_docs))
        pool.close()
        pool.join()
    else:
        for (counter,doc) in enumerate(raw_docs):     
            if doc != None:
                tokenized_docs.append(word_tokenize(doc))
            else:
                print('Doc '+str(counter+1)+' has None value!!!');
                tokenized_docs.append([]);
#             print tokenized_docs
    return tokenized_docs
    
# Step2: remove punctuation even they are valuable for tokenizing and save
# tokens in lower case  
def remove_punctiation(tokenized_docs = None, parallel = True, threads = 5):
    if tokenized_docs == None:
        raise Exception('Parameter error')
    
    print 'Removing punctiations!!...'
    tokenized_docs_no_punctuation = []
    
    if parallel == True:
        pool = Pool(processes = threads)
        tokenized_docs_no_punctuation = pool.map(remove_punctiation_foo, tokenized_docs, chunksize=100)
        pool.close()
        pool.join()
#        tokenized_docs_no_punctuation = Parallel(n_jobs=30, verbose=100, pre_dispatch=n_jobs/2)(delayed(remove_punctiation_foo) (doc) for doc in tokenized_docs)
    else:
        regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
        for review in tokenized_docs:
            new_review = []
            for token in review: 
                new_token = regex.sub(u'', token)
                if not new_token == u'':
                    new_review.append(new_token.lower())
            tokenized_docs_no_punctuation.append(new_review)

    return tokenized_docs_no_punctuation

# Dummy function for pooling
def remove_punctiation_foo(doc):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    new_doc = []        
    for token in doc: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_doc.append(new_token.lower())
    return new_doc        
 
    
# Step3: remove stop words
def remove_stopwords(tokenized_docs_no_punctuation = None, parallel = True, threads = 5):    
    if tokenized_docs_no_punctuation == None:
        raise Exception('Parameter error')
    
    filtered_docs = []
    if parallel == True:
        pool = Pool(processes = threads)     
        filtered_docs = pool.map(remove_stopword_from_file,tokenized_docs_no_punctuation)
    else:
        for doc in tokenized_docs_no_punctuation:
            new_term_vector = []
            for word in doc:
                if not word in stopwords.words('english'):
                    new_term_vector.append(word)
            filtered_docs.append(new_term_vector)
    
    return filtered_docs

# Dummy method for pooling 
def remove_stopword_from_file(doc):
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    return new_term_vector

# Step4: Strip words into roots to remove semantic duplicates
def perform_stem_lemma(tokenized_docs_no_stopwords):
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    
    preprocessed_docs = []
    
    for doc in tokenized_docs_no_stopwords:
        final_doc = []
        for word in doc:
            final_doc.append(porter.stem(word))
        preprocessed_docs.append(final_doc)
    
#     print preprocessed_docs
    return preprocessed_docs

# Perform all 4 steps above in one call
def perform_all_steps(docs=None):
    if docs == None:
        raise('Argument error!!!')
        
    print 'Documents are being preprocessed!!'
    docs = perform_tokenizing(docs)
    docs = remove_punctiation(docs)
    docs = remove_stopwords(docs)
    docs = perform_stem_lemma(docs);
    return docs

# if raw data is sourced from web this is a necessary to get rid of html tags
def remove_html(text):
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)

# Remove tags from list of documents
def remove_html_of_docs(docs):
    if docs == None:
        raise Exception('Parameter error')
    for (counter,doc) in enumerate(docs):
        docs[counter] = remove_html(doc)
    return docs



################################################
###### VOCABULARY GENERATION METHODS
################################################

def count_tokens_from_chunk(chunks_path):
    path_list = get_chunk_paths(chunks_path)
    tf = Counter()
    for counter,path in enumerate(path_list):
        print 'File : '+ str(counter)
        print 'Tokens : '+str(len(tf.keys()))
        f=open(path,'r')
        docs = cPickle.load(f)
        f.close()
        tf.update(count_token_from_doc_list(docs))                
    return zip(tf.keys(),tf.values())     

def count_token_from_doc_list(tokenized_doc_list):
    if tokenized_doc_list == None        :
        raise Exception('Parameter error')
        
    print 'Tokens are being counting!!!'  
    tf = Counter()
    
    pool = Pool(processes=5)
    results = pool.map(count_tokens, tokenized_doc_list)
    pool.close() 
#    results = Parallel(n_jobs=5, verbose = 5)(delayed(count_tokens) (doc) for doc in tokenized_doc_list)
    for result in results:
        tf.update(result)
    return tf   
    
# dummy function for pooling    
def count_tokens(doc):
    tf = Counter()    
    for token in doc:
        tf[token]+=1 
    return tf
            
# Create vocab from token count list by using some of the preferences
def filter_vocab(vocab = None, ratio = 1, return_counts = True, min_df = 1):
    if vocab == None:
        raise Exception('token count list should be given!!!')
    print 'Vocabulary is being created!!\n'
    
    # remove tokens with frequency smaller than min_df
    for item in [tup for tup in vocab if tup[1] < min_df]:
        vocab.remove(item)
    
    # sort tokens frequency and extract desired portion as vocab
    vocab = sorted(vocab,key = lambda x: x[1], reverse=True)
    size = len(vocab)
    num_tokens_of_interest = size*ratio
    offset = (size-num_tokens_of_interest)/2
    if not return_counts:
        return [item[0] for item in vocab[int(offset):int(size+offset)]]
    else:
        return vocab[int(offset):int(size+offset)]



################################################
###### CREATE BOW FEATURES 
################################################


def create_word_tag_matrix(tag_list_path, doc_chunks_path, output_path_data, processes_num = 20, merge_processes_num = 10):
    
#    tag_list        = cPickle.load(open(tag_list_path,'r'))        
    process_list    = [None]*processes_num
    counter_queue   = Queue() 
    merge_counter_queue = Queue()
#    word_tag_dict   = Counter()
    path_list       = get_chunk_paths(doc_chunks_path)    
    tag_path_list   = get_chunk_paths(tag_list_path)   
    num_files       = len(path_list)    
    
    
    if len(tag_path_list) !=  len(path_list):
        raise Exception('Given argument problem !!!')
        
#    for counter in range(doc_list.shape[0]):
#        print 'Document ' + str(counter+1) + ' of ' +str(doc_list.shape[0])
#        doc = list(extractor.inverse_transform(doc_list[counter]))[0]
#        tags = tag_list[counter]
#        for token in doc:
#            for tag in tags:
#                key = token+' '+tag
#                word_tag_dict[key] += 1
    file_counter = 0
    while_flag = True
    merging_processes = [Process(target=counter_merger, args=(counter_queue,merge_counter_queue,output_path_data, i, merge_processes_num)) for i in range(merge_processes_num)]
    
    # start counter merging processes    
    for process in merging_processes:    
        process.start()
        
    while while_flag:  
        process_list = []
        for process_no in range(processes_num):
            if file_counter == num_files:
                while_flag = False
                break 
            next_file_path = path_list[file_counter]
            next_tag_file_path = tag_path_list[file_counter]
            process_list.append(Process(target=count_word_tag_matrix, args=(next_tag_file_path, next_file_path, counter_queue, process_no)))
            file_counter = file_counter + 1
        
        print('File counter: ' +str(file_counter) )
        
        for process in process_list:
            process.start()
            
        for process in process_list:
            process.join()
        
        if while_flag == False:
            for process in merging_processes:
                counter_queue.put(-1) # stop signal
            for process in merging_processes:
                process.join()

    
def counter_merger(counter_queue, merge_counter_queue, output_path_data, process_no, merge_processes):
    master_counter = Counter()
    while True:
        item = counter_queue.get()
#        print('Merging')
        if item == -1:
           cPickle.dump(master_counter, open(str(process_no) + output_path_data,'w'))
           merge_counter_queue.put(master_counter)
#           master_counter_queue.put(master_counter) 
           if process_no == 0:
               master_counter = Counter()
               print('Final MERGING!!!')
               for i in range(merge_processes):
                   master_counter = master_counter + merge_counter_queue.get()
                   cPickle.dump(master_counter,open('all_merged'+ output_path_data,'w'))
           return
        master_counter = master_counter + item
        print('Process '+str(process_no)+' Number of items in matrix ' + str(len(master_counter.keys())))
        
        
def count_word_tag_matrix(tag_file_path,doc_file_path,queue, process_no):
#    print str(process_no)+' started ---!!!'
    docs = cPickle.load(open(doc_file_path,'r'))
    tags = pn.read_csv(tag_file_path)
    tags = list(np.array(tags['Tags'])) 
    matrix = Counter()
    for counter,doc in enumerate(docs):    
        tag_list = tags[counter]
        for token in doc:
                for tag in tag_list.split():
                    key = token+' '+tag
                    matrix[key] += 1
    queue.put(matrix)
#    print str(process_no)+' finished!!!'
#    print('Number of items in matrix ' + str(len(l[1].keys())))
        

    
if __name__ == '__main__':

#    divide_data_into_chunks('DATA/Train_no_code.csv','DATA/train_chunks',10000)
#    master_preprocessing_call(dummy_all_preprocessing,'DATA/train_chunks_csv',60,'DATA/tokenized_title_chunks_dumps', 'Title')
#    create_word_tag_matrix('DATA/train_chunks_csv', 'DATA/tokenized_title_chunks_dumps', 'title_word_tag_matrix.data', processes_num = 100, merge_processes_num = 40)    
    
#    
     master_vocab_generation(dummy_token_count,path_chunks='DATA/tokenized_body_chunks_dumps' ,output_path_data='word_counts.data' , processes=60, merge_processes=30)
    
#    token_counts = count_tokens_from_chunk('DATA/tokenized_chunks_dumps')    
#    cPickle.dump(Vocabulary, open('Vocab_Body.data','w'))
    
#    merge_word_tag_matrix_files('Body_Word_Tag_Matrices')