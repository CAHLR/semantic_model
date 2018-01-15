import os
import sys
import getopt
import time
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

vectorfile = ''
rawfile = ''
blobcolumn = ''
outputfile = ''
vocabsize = 0
num_top_words = 10 # hardcode for now
use_idf = False
num_clusters = 5
num_epochs = 5

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h:v:r:b:k:e:i:')
except getopt.GetoptError:
    print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -b <blobcolumn> [-k <num_clusters> -e <num_epochs> -i <use_idf>]')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -b <blobcolumn> [-k <num_clusters> -e <num_epochs> -i <use_idf>]')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<blobcolumn> is a column in raw file that needs nltk processing')
        print('<num_clusters> is the number of clusters to bin the data into (default 5)')
        print('<num_epochs> is the number of epochs to train the logistic regression model for (default 5)')
        print('<use_idf> is either True or False (default) for using idf')
        sys.exit()
    if opt in ("-v"):
        vectorfile = arg
    if opt in ("-r"):
        rawfile = arg
    if opt in ("-b"):
        blobcolumn = str(arg)
    if opt in ("-k"):
        num_clusters = int(arg)
    if opt in ("-e"):
        num_epochs = int(arg)
    if opt in ("-i"):
        use_idf = arg
if vectorfile == '':
    print('option [-v] must be set\n')
    sys.exit()
if rawfile == '':
    print('option [-r] must be set\n')
    sys.exit()

print('Vector input: ' + vectorfile)
print('Raw input: ' + rawfile)
print('Blob column: ' + blobcolumn)
outputfile = re.split("\.[a-z]{1,4}", rawfile)[0]+'_semantic__'+str(num_epochs)+'epochs'+str(num_clusters)+'clusters'+str(use_idf)
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
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_features=2500, use_idf=use_idf)
    X = vectorizer.fit_transform(dataframe[column])
    unigrams = vectorizer.get_feature_names()

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2,2), max_features=max(1, int(len(unigrams)/10)), use_idf=use_idf)
    X = vectorizer.fit_transform(dataframe[column])
    bigrams = vectorizer.get_feature_names()

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3,3), max_features=max(1, int(len(unigrams)/10)), use_idf=use_idf)
    X = vectorizer.fit_transform(dataframe[column])
    trigrams = vectorizer.get_feature_names()

    vocab = np.concatenate((unigrams, bigrams, trigrams))
    pd.DataFrame(vocab).to_csv(outputfile+'_vocab.tsv', sep = '\t', encoding='utf-8', index = False)
    return vocab

def to_bag_of_words(dataframe, column, vocab):
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocab, use_idf=False)
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
    model.fit(X, Y, epochs=num_epochs)
    weights = model.layers[1].get_weights()[0]
    biases = model.layers[1].get_weights()[1]
    weights_frame = pd.DataFrame(weights)
    biases_frame = pd.DataFrame(biases)
    weights_frame.to_csv(outputfile+'_weights.tsv', sep = '\t', index = False)
    biases_frame.to_csv(outputfile+'_biases.tsv', sep = '\t', index = False)
    return(weights_frame, biases)

def cluster(X):
    print('Clustering vectors...')
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi.
    # Add small number to denom to prevent nan.
    # ?? Can this be improved
    X = np.exp(-X / (X.std() + 1e-6))
    # ?? Online forums suggest arpack is more robust than amg
    sc = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack")
    return sc.fit_predict(X)

# MAIN
timebf = time.time()

# get data
vec_frame = read_big_csv(vectorfile)
len_vec_frame = len(vec_frame.index)
raw_frame = read_big_csv(rawfile)
len_raw_frame = len(raw_frame.index)

if (len_vec_frame != len_raw_frame):
    print('vector file and raw file entries do not line up\n')
    print(len_vec_frame, len_raw_frame)
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
    softmax_frame = vec_frame.iloc[:,1:].dot(weights_frame.values) + biases

    print('Sorting classification results...')
    sorted_frame = np.argsort(softmax_frame,axis=1).iloc[:,-num_top_words:]
    for i in range(num_top_words):
        new_col = vocab_frame.iloc[sorted_frame.iloc[:,i],0] # get the ith top vocab word for each entry
        raw_frame['predicted_word_' + str(num_top_words-i)] = new_col.values
    bow_frame = pd.DataFrame(M)

    # Assign each point to a cluster
    softmax_clusters = cluster(softmax_frame)
    # labels = cluster(raw_frame.iloc[:,1:3])
    raw_frame['softmax_cluster'] = softmax_clusters

    silhouette_avg = silhouette_score(vec_frame.iloc[:,1:], softmax_clusters, metric='cosine')
    print('Score for ' + str(num_clusters) + ' clusters, softmax input, vector eval result: ' + str(silhouette_avg))
    silhouette_avg = silhouette_score(raw_frame.iloc[:,1:3], softmax_clusters, metric='cosine')
    print('Score for ' + str(num_clusters) + ' clusters, softmax input, 2d eval result: ' + str(silhouette_avg))

    bow_clusters = cluster(bow_frame)
    raw_frame['bow_cluster'] = bow_clusters

    silhouette_avg = silhouette_score(vec_frame.iloc[:,1:], bow_clusters, metric='cosine')
    print('Score for ' + str(num_clusters) + ' clusters, bow input, vector eval result: ' + str(silhouette_avg))
    silhouette_avg = silhouette_score(raw_frame.iloc[:,1:3], bow_clusters, metric='cosine')
    print('Score for ' + str(num_clusters) + ' clusters, bow input, 2d eval result: ' + str(silhouette_avg))

print('Writing results to file...')
raw_frame.to_csv(outputfile+'.tsv', sep = '\t', index = False)
bow_frame.to_csv(outputfile+'_bow.tsv', sep = '\t', index = False)

print('Copying tsv files to txt for gzip')
os.system('mv '+ outputfile+'_weights.tsv' + ' ' + outputfile+'_weights.txt')
os.system('mv '+ outputfile+'_biases.tsv' + ' ' + outputfile+'_biases.txt')
os.system('mv '+ outputfile + '_vocab.tsv' + ' ' + outputfile + '_vocab.txt')
os.system('mv '+ outputfile+'.tsv' + ' ' + outputfile+'.txt')
os.system('mv '+ outputfile+'_bow.tsv' + ' ' + outputfile+'_bow.txt')
os.system('mv '+ vectorfile + ' ' + re.split("\.t.{0,3}", vectorfile)[0]+'.txt')
timeaf = time.time()
print('TIME: ', timeaf-timebf)
