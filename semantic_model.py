import subprocess
import sys
import getopt
import time
# import timeout_decorator
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from functools import wraps
import errno
import os
import signal

# Can use timeout decorator package: https://pypi.python.org/pypi/timeout-decorator
# Timeout decorator setup. See link for description.

# https://stackoverflow.com/questions/31822190/how-does-the-timeouttimelimit-decorator-work?noredirect=1&lq=1
class TimeoutError(Exception):
    pass

def timeout(seconds=100, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


vectorfile = ''
rawfile = ''
textcolumn = ''
outputfile = ''
cluster_input = 'both'
cluster_eval = 'both'

vocabsize = 0
num_top_words = 10 # hardcode for now
use_idf = False
num_clusters = 5
tf_bias = 1
num_epochs = 5
write_directory = './'
scorefile = './scorefile.txt'

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h:v:r:t:d:s:k:b:e:i:n:l:')
except getopt.GetoptError:
    print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -t <textcolumn> [-d <write_directory> (if not current directory) -s <scorefile_location> (if not ./scorefile.txt) -k <num_clusters> -b <tf_bias> -e <num_epochs> -i <use_idf> -n <cluster_input> -l <cluster_eval>]')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -t <textcolumn> [-d <write_directory> (if not current directory) -s <scorefile_location> (if not ./scorefile.txt) -k <num_clusters> -b <tf_bias> -e <num_epochs> -i <use_idf> -n <cluster_input> -l <cluster_eval>]')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<textcolumn> is a column in raw file that needs nltk processing')
        print('write_directory is where you would like to save the output files')
        print('scorefile_location is where you would like the scores to be saved')
        print('<num_clusters> is the number of clusters to bin the data into (default 5)')
        print('<tf_bias> is the bias constant for term-frequency (not used yet)')
        print('<num_epochs> is the number of epochs to train the logistic regression model for (default 5)')
        print('<use_idf> is either True or False (default) for using idf')
        print('<cluster_input> is either softmax, bow, or both (default)')
        print('<cluster_eval> is either vector, 2d, or both (default)')
        sys.exit()
    if opt in ("-v"):
        vectorfile = arg
    if opt in ("-r"):
        rawfile = arg
    if opt in ("-t"):
        textcolumn = arg
    if opt in ("-d"):
        write_directory = arg
    if opt in ("-s"):
        scorefile = arg
    if opt in ("-k"):
        num_clusters = int(arg)
    if opt in ("-b"):
        tf_bias = float(arg)
    if opt in ("-e"):
        num_epochs = int(arg)
    if opt in ("-i"):
        use_idf = arg
    if opt in ("-n"):
        cluster_input = arg
    if opt in ("-l"):
        cluster_eval = arg
if vectorfile == '':
    print('[DEBUG] option [-v] must be set\n')
    sys.exit()
if rawfile == '':
    print('[DEBUG] option [-r] must be set\n')
    sys.exit()

print('[INFO] Vector input: ' + vectorfile)
print('[INFO] Raw input: ' + rawfile)
print('[INFO] Text column: ' + textcolumn)
outputfile = re.split("\.t[a-z]{2}$", rawfile)[0]+'_semantic__'+str(num_epochs)+'epochs'+str(num_clusters)+'clusters'+str(use_idf)
outputfilename = outputfile.split('/')[-1]
print('[INFO] Output file: ' + outputfile)
print('[INFO] Output directory: ' + write_directory)

# Start
@timeout(600, "Timeout at read_big_csv.")
# @timeout_decorator.timeout(600, exception_message='timeout occured at read_big_csv')
def read_big_csv(inputfile):
    print('[INFO] Reading '+inputfile+'...')
    with open(inputfile,'r', encoding='utf-8') as f:
        a = f.readline()
    csvlist = a.split(',')
    tsvlist = a.split('\t')
    if len(csvlist)>len(tsvlist):
        sep = ','
    else:
        sep = '\t'
    print('[INFO] sep:' , sep)
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

@timeout(1200, "Timeout at get_vocab.")
# @timeout_decorator.timeout(1200, exception_message='timeout occured at get_vocab')
def get_vocab(dataframe, column):
    print("[INFO] Getting vocab...")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    dataframe[column] = dataframe[column].fillna('')

    print('[INFO] Taking at most 2500 (most frequent) unigrams')
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

@timeout(2400, "Timeout at to_bag_of_words.")
# @timeout_decorator.timeout(2400, exception_message='timeout occured at to_bag_of_words')
def to_bag_of_words(dataframe, column, vocab):
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocab, use_idf=False)
    X = vectorizer.fit_transform(dataframe[column].values.astype('U'))
    print(X)
    print((X.multiply(1/X.count_nonzero())).power(-tf_bias))
    return (X.multiply(1/X.count_nonzero())).power(-tf_bias)
    # return X

@timeout(7200, "Timeout at logistic_regression.")
# @timeout_decorator.timeout(7200, exception_message='timeout occured at logistic_regression')
def logistic_regression(X, Y):
    print('[INFO] Performing logistic regression...')
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

@timeout(4800, "Timeout at cluster.")
# @timeout_decorator.timeout(4800, exception_message='timeout occured at cluster')
def cluster(X):
    print('[INFO] Clustering vectors...')
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi.
    # Add small number to denom to prevent nan.
    # ?? Can this be improved
    # X = np.exp(-X / (X.std() + 1e-6))
    X = np.exp(- X**2 / (2. * 1 ** 2))
    clusters = None
    attempts = 0
    while clusters is None and attempts < 15:
        try:
            attempts += 1
            sc = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack")
            clusters = sc.fit_predict(X)
        except:
            print('[INFO] Clustering attempt # ' + str(attempts) + ' failed. ' + str(time.time()))
            pass
    if clusters is None:
        sys.exit('[ERROR] Unable to cluster in 15 attempts.')
    return clusters

# MAIN
timebf = time.time()
print('[INFO] Start time: ' + str(timebf))
# get data
time_get_data_bf = time.time()
vec_frame = read_big_csv(vectorfile)
raw_frame = read_big_csv(rawfile)

len_vec_frame = len(vec_frame.index)
len_raw_frame = len(raw_frame.index)
if (len_vec_frame != len_raw_frame):
    print('[DEBUG] vector file and raw file entries do not line up: ' + str(len_vec_frame) + ' ' + str(len_raw_frame))
    sys.exit()
time_get_data_af = time.time()
print('[INFO] getting data took ' + str(time_get_data_af - time_get_data_bf))

if (textcolumn != ''):
    ### Using the textcolumn, obtain a bow encoding and train the vectorspace coeffs to predict the bow of a point. ###
    time_get_vocab_and_bow_bf = time.time()
    vocab = get_vocab(raw_frame, textcolumn)
    vocab_frame = pd.DataFrame(vocab)
    vocabsize = len(vocab)
    # Convert the textcolumn of the raw dataframe into bag of words representation
    X = to_bag_of_words(raw_frame, textcolumn, vocab)
    M = X.toarray()
    bow_frame = pd.DataFrame(M)
    time_get_vocab_and_bow_af = time.time()
    print('[INFO] getting vocab and bow took ' + str(time_get_vocab_and_bow_af - time_get_vocab_and_bow_bf))

    # Train the coefficients for the vectorspace factors to predict the bag of words
    time_train_model_bf = time.time()
    (weights_frame, biases) = logistic_regression(vec_frame.iloc[:,1:], M)
    # Obtain the softmax predictions
    softmax_frame = vec_frame.iloc[:,1:].dot(weights_frame.values) + biases
    time_train_model_af = time.time()
    print('[INFO] training model took ' + str(time_get_data_af - time_get_data_bf))

    # From the softmax predictions, save the top 10 predicted words for each data point
    time_get_top_predictions_bf = time.time()
    print('[INFO] Sorting classification results...')
    sorted_frame = np.argsort(softmax_frame,axis=1).iloc[:,-num_top_words:]
    for i in range(num_top_words):
        new_col = vocab_frame.iloc[sorted_frame.iloc[:,i],0] # get the ith top vocab word for each entry
        raw_frame['predicted_word_' + str(num_top_words-i)] = new_col.values
    time_get_top_predictions_af = time.time()
    print('[INFO] getting top predictions for each point took ' + str(time_get_top_predictions_af - time_get_top_predictions_bf))

    ### Assign each point to a cluster based on the bag of word predictions ###
    time_assign_clusters_bf = time.time()
    softmax_clusters = cluster(softmax_frame)
    # Save cluster assignments to dataframe
    raw_frame['softmax_cluster'] = softmax_clusters
    time_assign_clusters_af = time.time()
    print('[INFO] assigning clusters took ' + str(time_assign_clusters_af - time_assign_clusters_bf))

    # Assess cluster assignments using vector cosine proximity
    time_cluster_score_bf = time.time()
    sf = open(scorefile,'a+')
    # if (cluster_input == 'softmax' or cluster_input == 'both'):
    if (cluster_input != 'bow'):
        # if (cluster_eval == 'vector' or 'both'):
        if (cluster_eval != '2d'):
            silhouette_avg = silhouette_score(vec_frame.iloc[:,1:], softmax_clusters, metric='cosine')
            print('Score--' + outputfilename + '\tsoftmax\tvector\t' + str(silhouette_avg))
            sf.write('\n' + outputfilename + '\tsoftmax\tvector\t' + str(silhouette_avg))
        # Assess cluster assignments using 2d vector cosine proximity
        # if (cluster_eval == '2d' or 'both'):
        if (cluster_eval != 'vector'):
            silhouette_avg = silhouette_score(raw_frame.iloc[:,1:3], softmax_clusters, metric='cosine')
            print('Score--' + outputfilename + '\tsoftmax\t2d\t' + str(silhouette_avg))
            sf.write('\n' + outputfilename + '\tsoftmax\t2d\t' + str(silhouette_avg))

    ### Repeat with actual bag of words ###
    # if (cluster_input == 'bow' or cluster_input == 'both'):
    if (cluster_input != 'softmax'):
        bow_clusters = cluster(bow_frame)
        raw_frame['bow_cluster'] = bow_clusters
        # if (cluster_eval == 'vector' or 'both'):
        if (cluster_eval != '2d'):
            silhouette_avg = silhouette_score(vec_frame.iloc[:,1:], bow_clusters, metric='cosine')
            print('Score--' + outputfilename + '\tbow\tvector\t' + str(silhouette_avg))
            sf.write('\n' + outputfilename + '\tbow\tvector\t' + str(silhouette_avg))
        # if (cluster_eval == '2d' or 'both'):
        if (cluster_eval != 'vector'):
            silhouette_avg = silhouette_score(raw_frame.iloc[:,1:3], bow_clusters, metric='cosine')
            print('Score--' + outputfilename + '\tbow\t2d\t' + str(silhouette_avg))
            sf.write('\n' + outputfilename + '\tbow\t2d\t' + str(silhouette_avg))
    sf.close()
    time_cluster_score_af = time.time()
    print('[INFO] calculating clustering scores took ' + str(time_cluster_score_af - time_cluster_score_bf))

print('[INFO] Writing results to file...')
time_writing_files_bf = time.time()
raw_frame.to_csv(outputfile+'.tsv', sep = '\t', index = False)
bow_frame.to_csv(outputfile+'_bow.tsv', sep = '\t', index = False)
time_writing_files_af = time.time()
print('[INFO] writing new raw_frame and bow_frame to file took ' + str(time_writing_files_af - time_writing_files_bf))

print('[INFO] Moving tsv files to txt for gzip')
time_txt_files_bf = time.time()
subprocess.call('mv '+ outputfile+'_weights.tsv' + ' ' + outputfile+'_weights.txt', shell=True)
subprocess.call('mv '+ outputfile+'_biases.tsv' + ' ' + outputfile+'_biases.txt', shell=True)
subprocess.call('mv '+ outputfile + '_vocab.tsv' + ' ' + outputfile + '_vocab.txt', shell=True)
subprocess.call('mv '+ outputfile+'.tsv' + ' ' + outputfile+'.txt', shell=True)
subprocess.call('mv '+ outputfile+'_bow.tsv' + ' ' + outputfile+'_bow.txt', shell=True)
subprocess.call('cp '+ vectorfile + ' ' + re.split("\.t.{0,3}", vectorfile)[0]+'.txt', shell=True)

subprocess.call('mv '+ outputfile+'_weights.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile+'_biases.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile + '_vocab.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile+'.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile+'_bow.txt ' + write_directory, shell=True)
subprocess.call('mv '+ re.split("\.t.{0,3}", vectorfile)[0]+'.txt ' + write_directory, shell=True)
time_txt_files_af = time.time()
print('[INFO] Moving tsv files to txt took ' + str(time_txt_files_af - time_txt_files_bf))
# subprocess.call('mv VS-*.txt ' + write_directory, shell=True)
# subprocess.call('mv *semantic*.txt ' + write_directory, shell=True)
timeaf = time.time()
print('[INFO] End time: ' + str(timeaf))
print('[INFO] TOTAL time:', timeaf-timebf)
