# import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
import nltk
import string

vectorfile = ''
rawfile = ''
blobcolumn = ''
outputfile = ''

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
    print("reading data...")
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

def process_text(dataframe, column):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer

    dataframe[column] = dataframe[column].fillna('')
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    X = vectorizer.fit_transform(dataframe[column])
    vocab = vectorizer.get_feature_names()
    write_vocab_file(vocab)
    return X

def write_vocab_file(vocab):
    vocabfile = open('vocab.tsv', 'w')
    for item in vocab:
        vocabfile.write("%s\n" % item)

# main

# timebf = time.time()
# hiD vector file
vec_frame = read_big_csv(vectorfile)
len_vec_frame = len(vec_frame.index)
# raw file
raw_frame = read_big_csv(rawfile)
len_raw_frame = len(raw_frame.index)
if (len_vec_frame != len_raw_frame):
    print('vector file and raw file entries do not line up\n')
    sys.exit()

if (blobcolumn != ''):
    X = process_text(raw_frame, blobcolumn)
    M = X.toarray()
    for index, row in raw_frame.iterrows():
        raw_frame.set_value(index, blobcolumn, M[index])
    vec_frame['bow'] = list(X.toarray())
    # from nltk.corpus import stopwords
    # from nltk.tokenize import word_tokenize
    # from sklearn.feature_extraction.text import CountVectorizer

    # raw_frame[blobcolumn] = raw_frame[blobcolumn].fillna('')
    # print(raw_frame[blobcolumn])
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    # X = vectorizer.fit_transform(raw_frame[blobcolumn])
    # M = X.toarray()
    # for index, row in raw_frame.iterrows():
    #     raw_frame.set_value(index, blobcolumn, M[index])
    # print(vec_frame.shape)
    # print(X.shape)
    # vec_frame['bow'] = list(X.toarray())
    # vocab = vectorizer.get_feature_names()
    # print(type(vocab))
    # vocabfile = open('vocab.tsv', 'w')
    # for item in vocab:
    #     vocabfile.write("%s\n" % item)
# timeaf = time.time()
# print('TIME: ',timeaf-timebf)
raw_frame.to_csv(outputfile, sep = '\t', index = False)
vec_frame.to_csv('bow_'+vectorfile, sep = '\t', index = False)

# frame = pd.merge(frame,  feature_frame, how = 'left', on = frame.columns[0])
# frame.to_csv(outputfile, sep = '\t', index = False)
