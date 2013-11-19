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
###### - PARALLEL CARDINALITY ESTIMATION - This part cen be coded fully recursive manner!!
################################################

# GIVEN THE PROCESS FUNCTION AND THE MERGE FUNCTIONS PERFORM PARALLEL EXECUTION OF COUNTING 
# AND MERGING OF COUNTERS  
def master_parallel_run(process_function, merger_function , path_chunks=None 
    ,output_path_data=None, no_procs=5, no_merge_procs = 5, tag_file_path=None):

    if path_chunks ==  None:
            raise Exception('Path should be defined')
        
    counter_queue   = Queue() 
    merge_counter_queue = Queue()

    process_list    = [None]*no_procs
    merging_processes = [Process(target=merger_function, args=(counter_queue,merge_counter_queue,output_path_data, i, no_merge_procs)) for i in range(no_merge_procs)]

    path_list       = get_chunk_paths(path_chunks)
    
    if tag_file_path != None:    
        tag_path_list   = get_chunk_paths(tag_file_path)   

    num_files       = len(path_list)    
    
    for process in merging_processes:    
        process.start()    
        
    file_counter = 0;
    while_flag = True
    while while_flag:
        process_list = []
        for process_no in range(no_procs):
            
            if file_counter == num_files:
                while_flag = False
                break       
            next_path = path_list[file_counter]
#            docs_list[process_no] = list(np.array(pn.read_csv(next_path)['Body']))
#            docs_list[process_no] = cPickle.load(open(next_path,'r'))
            if tag_file_path == None:
                process_list.append(Process(target=process_function, args= (next_path, process_no, counter_queue)))
            else:
                process_list.append(Process(target=process_function, args= (tag_file_path, next_path, process_no, counter_queue)))

            file_counter = file_counter+1
        
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
       
        
        
def merge_counters(counter_queue, merge_counter_queue, output_path_data, process_no, no_merge_procs):
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
               cPickle.dumps(master_counter, f)
               print("FINISHED!!!")
               return
        master_counter = master_counter + item
        print('Process '+str(process_no)+' Number of items in matrix ' + str(len(master_counter.keys())))
    
# Count the occurrance of tokens in the given data chunks
def token_count(chunk_file, process_no, queue):
    # print str(process_no)+' started --- Token counting!!!'
    tf = Counter()
    f = open(chunk_file,'r')
    docs = cPickle.load(f)
    f.close()
    if docs == None        :
        raise Exception('Doc list should be given !!!')  
    for counter,doc in enumerate(docs):
        try:
            for token in doc:
                tf[token] += 1
        except:
            continue
    queue.put(tf)
    
# Count the word_tag co-occurrances
def count_word_tag_matrix(tag_file_path,doc_file_path, process_no, queue):
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


if __name__ == '__main__':

#    divide_data_into_chunks('DATA/Train_no_code.csv','DATA/train_chunks',10000)
#    master_preprocessing_call(dummy_all_preprocessing,'DATA/train_chunks_csv',60,'DATA/tokenized_title_chunks_dumps', 'Title')
#    create_word_tag_matrix('DATA/train_chunks_csv', 'DATA/tokenized_title_chunks_dumps', 'title_word_tag_matrix.data', processes_num = 100, no_merge_procs = 40)    
    
#    
     master_parallel_run(token_count, merge_counters, path_chunks='DATA/tokenized_body_chunks_dumps' 
        ,output_path_data='word_counts.data' , no_procs=20, no_merge_procs=50)
    
#    token_counts = count_tokens_from_chunk('DATA/tokenized_chunks_dumps')    
#    cPickle.dump(Vocabulary, open('Vocab_Body.data','w'))
    
#    merge_word_tag_matrix_files('Body_Word_Tag_Matrices')
