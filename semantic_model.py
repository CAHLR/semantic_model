import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
import string


vectorfile = ''
rawfile = ''
blobcolumn = ''
outputfile = ''
vocabsize = 0
num_top_words = 10 # hardcode for now

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h:v:r:b:')
except getopt.GetoptError:
    print('\nxxx.py -v <vectorfile> -r <rawfile> -b <blobcolumn>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\nxxx.py -v <vectorfile> -r <rawfile> -b <blobcolumn>')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<blobcolumn> is a column in raw file that needs nltk processing')
        sys.exit()
    if opt in ("-v"):
        vectorfile = arg
    if opt in ("-r"):
        rawfile = arg
        outputfile = re.split("\.[a-z]{1,4}", rawfile)[0]+'_semantic'
    if opt in ("-b"):
        blobcolumn = str(arg)
if vectorfile == '':
    print('option [-v] must be set\n')
    sys.exit()
if rawfile == '':
    print('option [-r] must be set\n')
    sys.exit()
if outputfile == '':
    print('option [-o] must be set\n')
    sys.exit()

print('Vector input: ' + vectorfile)
print('Raw input: ' + rawfile)
print('Blob column: ' + blobcolumn)
print('Output file: ' + outputfile)

# Start

def read_big_csv(inputfile):
    print("Reading "+inputfile+"...")
    with open(inputfile,'r') as f:
        a = f.readline()
    csvlist = a.split(',')
    tsvlist = a.split('\t')
    if len(csvlist)>len(tsvlist):
        sep = ','
    else:
        sep = '\t'
    print('sep:' , sep)
    reader = pd.read_csv(inputfile, iterator=True, low_memory = False,delimiter=sep)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    df = pd.concat(chunks, ignore_index=True)
    return df

def get_vocab(dataframe, column):
    print("Getting vocab...")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    dataframe[column] = dataframe[column].fillna('')
    print('Taking at most 2500 unigrams')
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1), max_features=2500)
    X = vectorizer.fit_transform(dataframe[column])
    unigrams = vectorizer.get_feature_names()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2), max_features=int(len(unigrams)/10))
    X = vectorizer.fit_transform(dataframe[column])
    bigrams = vectorizer.get_feature_names()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(3,3), max_features=int(len(unigrams)/100))
    X = vectorizer.fit_transform(dataframe[column])
    trigrams = vectorizer.get_feature_names()

    vocab = np.concatenate((unigrams, bigrams, trigrams))
    pd.DataFrame(vocab).to_csv(outputfile+'_vocab.tsv', sep = '\t', encoding='utf-8', index = False)
    return vocab

def to_bag_of_words(dataframe, column, vocab):
    vectorizer = CountVectorizer(stop_words='english', vocabulary=vocab)
    X = vectorizer.fit_transform(dataframe[column].values.astype('U'))
    return X

def logistic_regression(X, Y):
    print('Performing logistic regression...')
    from keras.layers import Input, Dense
    from keras.models import Model
    inputs = Input(shape=(X.shape[1],))
    predictions = Dense(vocabsize, activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(X, Y)
    weights = model.layers[1].get_weights()[0]
    biases = model.layers[1].get_weights()[1]
    weights_frame = pd.DataFrame(weights)
    biases_frame = pd.DataFrame(biases)
    weights_frame.to_csv(outputfile+'_weights.tsv', sep = '\t', index = False)
    biases_frame.to_csv(outputfile+'_biases.tsv', sep = '\t', index = False)
    return(weights_frame, biases)

# main
timebf = time.time()

# get data
vec_frame = read_big_csv(vectorfile)
len_vec_frame = len(vec_frame.index)
raw_frame = read_big_csv(rawfile)
len_raw_frame = len(raw_frame.index)

if (len_vec_frame != len_raw_frame):
    print('vector file and raw file entries do not line up\n')
    sys.exit()

if (blobcolumn != ''):
    # Recompute vocab every time, just in case.
    vocab = get_vocab(raw_frame, blobcolumn)
    vocab_frame = pd.DataFrame(vocab)
    vocabsize = len(vocab)
    X = to_bag_of_words(raw_frame, blobcolumn, vocab)
    M = X.toarray()
    # Recompute coefficients every time, just in case.
    (weights_frame, biases) = logistic_regression(vec_frame.iloc[:,1:], M)
    result_frame = vec_frame.iloc[:,1:].dot(weights_frame.values)
    result_frame += biases
    print('Sorting classification results...')
    sorted_frame = np.argsort(result_frame,axis=1).iloc[:,-num_top_words:]
    for i in range(num_top_words):
        new_col = vocab_frame.iloc[sorted_frame.iloc[:,i],0] # get the ith top vocab word for each entry
        raw_frame['predicted_word_' + str(num_top_words-i)] = new_col.values

    bow_frame = pd.DataFrame(M)

print('Writing results to file...')
raw_frame.to_csv(outputfile+'.tsv', sep = '\t', index = False)
bow_frame.to_csv(outputfile+'_bow.tsv', sep = '\t', index = False)

print('Copying tsv files to txt for gzip')
os.system('cp '+ outputfile+'_weights.tsv' + ' ' + outputfile+'_weights.txt')
os.system('cp '+ outputfile+'_biases.tsv' + ' ' + outputfile+'_biases.txt')
os.system('cp '+ outputfile + '_vocab.tsv' + ' ' + outputfile + '_vocab.txt')
os.system('cp '+ outputfile+'.tsv' + ' ' + outputfile+'.txt')
os.system('cp '+ outputfile+'_bow.tsv' + ' ' + outputfile+'_bow.txt')
os.system('cp '+ vectorfile + ' ' + re.split(".tsv", vectorfile)[0]+'.txt')
timeaf = time.time()
print('TIME: ',timeaf-timebf)
