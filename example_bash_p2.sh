#!/bin/bash


filebaselist=$1 # e.g. rp_output.txt, generated in directory that representation_presenter.py was called
outdir=$2 # e.g. ~/semantic-data/
clustercounts="5,10,15"
clusterinput=bow+softmax # bow+softmax+pass when clustercounts = 2 only
clustereval=vector+2d
datadir="~/symlink/" # where the representation presenter results are
tfbias="-999"

# Uncomment for command line control
#clustercounts=$3 # e.g. 5,10
#clusterinput=$4 # e.g. softmax
#clustereval=$5 # e.g. vector
#tfbias=$6 # e.g. -999
#datadir=$7

score_recordfile=${outdir}scorefile$(date +%Y-%m-%d-%T).txt
> $score_recordfile
while read filebase
do
    IFS=","
    for numclusters in $clustercounts
    do
        echo "Running semantic_model for ${filebase} with ${numclusters} clusters"
        echo "echo python3 ~/semantic_model/semantic_model.py -v ${datadir}VS-${filebase} -r ${datadir}${filebase} -t goals -d ${outdir} -s ${score_recordfile} -e 20 -k ${numclusters} -n ${clusterinput} -l ${clustereval} -b ${tfbias}"
        echo python3 ~/semantic_model/semantic_model.py -v ${datadir}VS-${filebase} -r ${datadir}${filebase} -t goals -d ${outdir} -s ${score_recordfile} -e 20 -k ${numclusters} -n ${clusterinput} -l ${clustereval} -b ${tfbias} | qsub
    done
done < $filebaselist
