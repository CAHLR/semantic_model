import sys
import getopt
import time
import pandas as pd
import numpy as np

weightsfile = ''
biasfile = ''
vocabfile = ''
outputfile = ''
datalist = []

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:v:p:o:')
except getopt.GetoptError:
    print('\nxxx.py -d <datalist> -v <vectorfile> -p <filesprefix> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('\nxxx.py -d <datalist> -w <weightsfile> -b <biasfile> -o <outputfile>')
        print('<datalist> is the comma-separated list of indices for requested data (e.g. 1,4,23,22,1)')
        print('<vectorfile> is the high dimensional vector\n<rawfile> is the original input data')
        print('<filesprefix> is the prefix of the weights, biases, and vocab files')
        print('<outputfile> is the name of your output file')
        sys.exit()
    if opt in ("-d"):
        datalist = arg.split(',')
        datalist = [int(datum) for datum in datalist]
    if opt in ("-v"):
        vectorfile = arg
    if opt in ("-p"):
        filesprefix = arg
        weightsfile = filesprefix+'_weights.tsv'
        biasfile = filesprefix+'_biases.tsv'
        vocabfile = filesprefix+'_vocab.tsv'
    if opt in ("-o"):
        outputfile = arg

if len(datalist) == 0:
    print('option [-d] must be set\n')
    sys.exit()
if vectorfile == '':
    print('option [-v] must be set\n')
    sys.exit()
if weightsfile == '':
    print('option [-w] must be set\n')
    sys.exit()
if biasfile == '':
    print('option [-b] must be set\n')
    sys.exit()
if outputfile == '':
    print('option [-o] must be set\n')
    sys.exit()

print(datalist)
print('Vector input: ' + vectorfile)
print('Weights input: ' + weightsfile)
print('Bias input: ' + biasfile)
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

# main
timebf = time.time()

# get data
vec_frame = read_big_csv(vectorfile)
weights_frame = read_big_csv(weightsfile)
biases_frame = read_big_csv(biasfile).transpose()
vocab_frame = read_big_csv(vocabfile)
print('weights_frame', weights_frame)
print('biases_frame', biases_frame)
print('vocab_frame', vocab_frame)


# average the high dim'l vectors
for datum in datalist:
    if (datum > len(vec_frame) or datum == 0):
        print('invalid data index ' + datum)
        sys.exit()
vectors = vec_frame.iloc[datalist,1:]
print('start')

avg_vector = np.transpose(pd.DataFrame(vectors.mean()))
print(avg_vector)

mul = avg_vector.dot(weights_frame.values)
print('mul')
print(mul)
# result = np.array(mul.add(biases_frame.values).iloc[0,:])
result = mul.add(biases_frame.values).iloc[0,:]
print('add')
print(result)
print('b')
top_prob_words = result.argsort()[-10:][::-1]
print(top_prob_words)
for index in top_prob_words:
    print(index)
    print(vocab_frame.iloc[index], result.iloc[index])


timeaf = time.time()
print('TIME: ',timeaf-timebf)
