# import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import string


vectorfile = ''
rawfile = ''
blobcolumn = ''
outputfile = ''
vocabsize = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h:v:r:b:o:')
except getopt.GetoptError:
    print('\nxxx.py -v <vectorfile> -r <rawfile> -b <blobcolumn> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\nxxx.py -v <vectorfile> -r <rawfile> -b <blobcolumn> -o <outputfile>')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<blobcolumn> is a column in raw file that needs nltk processing\n<outputfile> is the name of your output file')
        sys.exit()
    if opt in ("-v"):
        vectorfile = arg
    if opt in ("-r"):
        rawfile = arg
    if opt in ("-b"):
        blobcolumn = str(arg)
    if opt in ("-o"):
        outputfile = arg
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
print('Blob column: ' + blobcolumn) # check later if cast is needed
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
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    dataframe[column] = dataframe[column].fillna('')
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1))
    X = vectorizer.fit_transform(dataframe[column])
    unigrams = vectorizer.get_feature_names()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2), max_features=int(len(unigrams)/10))
    X = vectorizer.fit_transform(dataframe[column])
    bigrams = vectorizer.get_feature_names()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(3,3), max_features=int(len(unigrams)/100))
    X = vectorizer.fit_transform(dataframe[column])
    trigrams = vectorizer.get_feature_names()

    vocab = np.concatenate((unigrams, bigrams, trigrams))
    write_vocab_file(vocab)
    return vocab


def to_bag_of_words(dataframe, column, vocab):
    vectorizer = CountVectorizer(stop_words='english', vocabulary=vocab)
    X = vectorizer.fit_transform(dataframe[column])
    return X

def write_vocab_file(vocab):
    vocabfile = open(outputfile+'_vocab.tsv', 'w')
    for item in vocab:
        vocabfile.write("%s\n" % item)

def logistic_regression(X, Y):
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
    pd.DataFrame(weights).to_csv(outputfile+'_weights.tsv', sep = '\t', index = False)
    pd.DataFrame(biases).to_csv(outputfile+'_biases.tsv', sep = '\t', index = False)

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
    print("Performing text processing...")
    vocab = get_vocab(raw_frame, blobcolumn)
    vocabsize = len(vocab)
    X = to_bag_of_words(raw_frame, blobcolumn, vocab)
    M = X.toarray()
    for index, row in raw_frame.iterrows():
        raw_frame.at[index, blobcolumn] = M[index]
    logistic_regression(vec_frame.iloc[:,1:], M)
    vec_frame['bow'] = list(M)
raw_frame.to_csv(outputfile+'.tsv', sep = '\t', index = False)
vec_frame.to_csv(outputfile+'_bow_'+vectorfile, sep = '\t', index = False)
timeaf = time.time()
print('TIME: ',timeaf-timebf)
