import subprocess
import os
import sys
import getopt
import time
# Timeout decorator package from: https://pypi.python.org/pypi/timeout-decorator
import timeout_decorator
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
textcolumn = ''
outputfile = ''
cluster_input = []
cluster_eval = []

vocabsize = 0
num_top_words = 10 # hardcode for now
use_idf = False
num_clusters = 5
tf_bias = -999
num_epochs = 5
write_directory = './'
scorefile = './scorefile.txt'
finishedfile = './finishedfile.txt'

predict = False


try:
    opts, args = getopt.getopt(sys.argv[1:], 'hv:r:t:d:s:f:k:b:e:i:p:n:l:')
except getopt.GetoptError:
    print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -t <textcolumn> [-d <write_directory> (if not current directory) -s <scorefile_location> (if not ./scorefile.txt) -f <finishedfile_location> (if not ./finishedfile.txt) -k <num_clusters> -b <tf_bias> -e <num_epochs> -i <use_idf> -p <predict> -n <cluster_input> -l <cluster_eval>]')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\npython3 semantic_model.py -v <vectorfile> -r <rawfile> -t <textcolumn> [-d <write_directory> (if not current directory) -s <scorefile_location> (if not ./scorefile.txt) -f <finishedfile_location> (if not ./finishedfile.txt) -k <num_clusters> -b <tf_bias> -e <num_epochs> -i <use_idf> -n <cluster_input> -l <cluster_eval>]')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<textcolumn> is a column in raw file that needs nltk processing')
        print('<write_directory> is where you would like to save the output files')
        print('<scorefile_location> is where you would like the scores to be saved')
        print('<finishedfile_location> is where you would like the scores to be saved')
        print('<num_clusters> is the number of clusters to bin the data into (default 5)')
        print('<tf_bias> is the bias constant for term-frequency')
        print('<num_epochs> is the number of epochs to train the logistic regression model for (default 5)')
        print('<use_idf> is either True or False (default) for using idf')
        print('<predict> is either True or False (default) for doing predictions on what empty entries might be')
        print('<cluster_input> is a '+'-separated (no spaces) list like softmax+bow (default)')
        print('<cluster_eval> is a '+'-separated (no spaces) list like vector+2d (default)')
        sys.exit()
    if opt in ("-v"):
        print('[INFO] setting -v')
        vectorfile = arg
    if opt in ("-r"):
        print('[INFO] setting -r')
        rawfile = arg
    if opt in ("-t"):
        print('[INFO] setting -t')
        textcolumn = arg
    if opt in ("-d"):
        print('[INFO] setting -d')
        write_directory = arg
    if opt in ("-s"):
        print('[INFO] setting -s')
        scorefile = arg
    if opt in ("-f"):
        print('[INFO] setting -f')
        finishedfile = arg
    if opt in ("-k"):
        print('[INFO] setting -k')
        num_clusters = int(arg)
    if opt in ("-b"):
        print('[INFO] setting -b')
        tf_bias = float(arg)
    if opt in ("-e"):
        print('[INFO] setting -e')
        num_epochs = int(arg)
    if opt in ("-i"):
        print('[INFO] setting -i')
        use_idf = arg
    if opt in ("-p"):
        print('[INFO] setting -p')
        predict = arg.lower()
    if opt in ("-n"):
        print('[INFO] setting -n')
        print(arg)
        print(len(arg.split("+")))
        cluster_input = arg.lower()
        cluster_input = cluster_input.split("+")
        print('[DEBUG] cluster inputs: ')
        print(cluster_input)
    if opt in ("-l"):
        print('[INFO] setting -l')
        cluster_eval = arg.lower()
        cluster_eval = cluster_eval.split("+")
        print('[DEBUG] cluster evals: ')
        print(cluster_eval)
         
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
if tf_bias != -999:
    print('[INFO] Term-frequency bias: ' + str(tf_bias))
    outputfile = outputfile + str(tf_bias)
print('[INFO] Clustering by: ' + ', '.join(cluster_input))
print('[INFO] Evaluating clusters by: ' + ', '.join(cluster_eval))
outputfilename = outputfile.split('/')[-1]
print('[INFO] Output file: ' + outputfile)
print('[INFO] Output directory: ' + write_directory)

# Start
@timeout_decorator.timeout(600, exception_message='timeout occured at read_big_csv')
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

@timeout_decorator.timeout(600, exception_message='timeout occured at get_vocab')
def get_vocab(dataframe, column):
    print("[INFO] Getting vocab...")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    dataframe[column] = dataframe[column].fillna('')

    print('[INFO] Taking at most 2000 (most frequent) unigrams')
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_features=2000, use_idf=use_idf)
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

@timeout_decorator.timeout(600, exception_message='timeout occured at to_bag_of_words')
def to_bag_of_words(dataframe, column, vocab):
    """Input: raw dataframe, text column, and vocabulary.
    Returns a sparse matrix of the bag of words representation of the column."""
    vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vocab, use_idf=False)
    X = vectorizer.fit_transform(dataframe[column].values.astype('U'))
    if tf_bias == -999:
        return X
    return (X.multiply(1/X.count_nonzero())).power(-tf_bias)

@timeout_decorator.timeout(3600, exception_message='timeout occured at logistic_regression')
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
    weights_frame.to_csv(outputfile+'_weights.txt', sep = '\t', index = False)
    subprocess.call('mv '+ outputfile+'_weights.txt ' + write_directory, shell=True)
    biases_frame.to_csv(outputfile+'_biases.txt', sep = '\t', index = False)
    subprocess.call('mv '+ outputfile+'_biases.txt ' + write_directory, shell=True)
    return(weights_frame, biases)

@timeout_decorator.timeout(3600, exception_message='timeout occured at cluster')
def cluster(X):
    print('[INFO] Clustering vectors...')
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi.
    # Add small number to denom to prevent nan.
    # ?? Can this be improved?
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
vec_frame = read_big_csv(vectorfile) # VS representation of each user, all numeric
raw_frame = read_big_csv(rawfile) # User information, various data

len_vec_frame = len(vec_frame.index)
len_raw_frame = len(raw_frame.index)
if (len_vec_frame != len_raw_frame):
    print('[DEBUG] vector file and raw file entries do not line up: ' + str(len_vec_frame) + ' ' + str(len_raw_frame))
    sys.exit()

nonempty_indices = np.where(raw_frame[textcolumn].notnull() == True)[0]
filtered_vec_frame = vec_frame.iloc[nonempty_indices,:]
filtered_raw_frame = raw_frame.iloc[nonempty_indices,:]

time_get_data_af = time.time()
print('[INFO] Getting data took ' + str(time_get_data_af - time_get_data_bf))

if (textcolumn != ''):
    ### Using the textcolumn, obtain a bow encoding and train the vectorspace coeffs to predict the bow of a point. ###
    # Get the vocab
    time_get_vocab_and_bow_bf = time.time()
    vocab = get_vocab(raw_frame, textcolumn)
    vocab_frame = pd.DataFrame(vocab)
    vocabsize = len(vocab)
    # Convert the textcolumn of the raw dataframe into bag of words representation
    bow_spmatrix = to_bag_of_words(raw_frame, textcolumn, vocab)
    bow_ndarray = bow_spmatrix.toarray()
    bow_frame = pd.DataFrame(bow_ndarray)

    filtered_bow_spmatrix = to_bag_of_words(filtered_raw_frame, textcolumn, vocab)
    filtered_bow_ndarray = filtered_bow_spmatrix.toarray()
    # filtered_bow_frame = pd.DataFrame(filtered_bow_ndarray)
    time_get_vocab_and_bow_af = time.time()
    print('[INFO] Getting vocab and bow took ' + str(time_get_vocab_and_bow_af - time_get_vocab_and_bow_bf))

    # Train the coefficients for the vectorspace factors to predict the bag of words
    time_train_model_bf = time.time()
    
    if predict:
        # Only train on instances with non-empty texts
        (weights_frame, biases) = logistic_regression(filtered_vec_frame.iloc[:,1:], filtered_bow_ndarray)
        # Obtain the softmax predictions for all instances
        softmax_frame = vec_frame.iloc[:,1:].dot(weights_frame.values) + biases
        time_train_model_af = time.time()
        print('[INFO] Training model took ' + str(time_get_data_af - time_get_data_bf))

        # From the softmax predictions, save the top 10 predicted words for each data point
        time_get_top_predictions_bf = time.time()
        print('[INFO] Sorting classification results...')
        sorted_frame = np.argsort(softmax_frame,axis=1).iloc[:,-num_top_words:]
        for i in range(num_top_words):
            new_col = vocab_frame.iloc[sorted_frame.iloc[:,i],0] # get the ith top vocab word for each entry
            raw_frame['predicted_word_' + str(num_top_words-i)] = new_col.values
        time_get_top_predictions_af = time.time()
        print('[INFO] Getting top predictions for each point took ' + str(time_get_top_predictions_af - time_get_top_predictions_bf))

    ### Cluster points based on various metrics and evaluating each clustering ###
    sf = open(scorefile,'a+')
    time_clusters_bf = time.time()
    for elem in cluster_input:
        print('[INFO] Clustering using ' + elem)
        if (elem == 'softmax'):
            if predict:
                continue 
            clusters = cluster(softmax_frame) + 1
        elif elem == 'bow':
            # Do not cluster those without any words
            clusters = np.zeros(len(raw_frame[textcolumn]), dtype=np.int)
            clusters[nonempty_indices] = cluster(filtered_bow_ndarray) + 1
        else:
            if (np.issubdtype(df[elem].dtype, np.number)):
                clusters = raw_frame[[elem]] + 1
            else:
                print('[ERROR] feature ' + elem + ' is not numeric')
        print('[DEBUG] Got clusters: ')
        print(clusters)
        # Save cluster assignments to dataframe
        raw_frame[elem + '_cluster'] = clusters
        # Evaluate the clustering's cosine proximity
        for el in cluster_eval:
            print('[INFO] Evaluating clustering using ' + el)
            if el == 'vector':
                eval_by = vec_frame.iloc[:,1:]
            elif el == '2d':
                eval_by = raw_frame[['x', 'y']]
            else:
                if (np.issubdtype(df[elem].dtype, np.number)):
                    eval_by = raw_frame[[el]]
                else:
                    print('[ERROR] feature ' + elem + ' is not numeric')
            if (elem == 'bow'):
		# Don't evaluate default 0 clusters
                silhouette_avg = silhouette_score(eval_by.iloc[nonempty_indices,:], clusters[nonempty_indices], metric='cosine')
            else:
                silhouette_avg = silhouette_score(eval_by, clusters, metric='cosine')
            print('Score--' + outputfilename + '\t'+elem+'\t'+el+'\t' + str(silhouette_avg))
            sf.write('\n' + outputfilename + '\t'+elem+'\t'+el+'\t' + str(silhouette_avg))
    time_clusters_af = time.time()
    print('[INFO] Assigning and evaluating clusters took ' + str(time_clusters_af - time_clusters_bf))
    print('[INFO] Scores available through logs or in ' + scorefile)
    sf.close()

print('[INFO] Writing results to file...')
time_writing_files_bf = time.time()
raw_frame.to_csv(outputfile+'.tsv', sep = '\t', index = False)
bow_frame.to_csv(outputfile+'_bow.tsv', sep = '\t', index = False)
time_writing_files_af = time.time()
print('[INFO] Writing new raw_frame and bow_frame to file took ' + str(time_writing_files_af - time_writing_files_bf))

# Moving tsv files to txt for gzip
time_txt_files_bf = time.time()
subprocess.call('mv '+ outputfile + '_vocab.tsv' + ' ' + outputfile + '_vocab.txt', shell=True)
subprocess.call('mv '+ outputfile+'.tsv' + ' ' + outputfile+'.txt', shell=True)
subprocess.call('mv '+ outputfile+'_bow.tsv' + ' ' + outputfile+'_bow.txt', shell=True)
subprocess.call('cp '+ vectorfile + ' ' + re.split("\.t.{0,3}", vectorfile)[0]+'.txt', shell=True)

subprocess.call('mv '+ outputfile + '_vocab.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile+'.txt ' + write_directory, shell=True)
subprocess.call('mv '+ outputfile+'_bow.txt ' + write_directory, shell=True)
subprocess.call('mv '+ re.split("\.t.{0,3}", vectorfile)[0]+'.txt ' + write_directory, shell=True)
time_txt_files_af = time.time()
timeaf = time.time()

ff = open(finishedfile, 'a+')
ff.write('\n' + outputfilename)
ff.close()
print('[INFO] End time: ' + str(timeaf))
print('[INFO] TOTAL time:', timeaf-timebf)
