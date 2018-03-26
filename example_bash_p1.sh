#!/bin/bash

# This script automates the following command on every dataset in specified directory
#python3 ~/representation_presenter/representation_presenter.py -i ~/event-data/${dataset} -o ${output}_${groupby}.tsv,4 -d ~/data/ -v ${vectorsize} -w ${windowsize} -t ~/bhtsne/ -g ${groupby} -k userid -s time -f ${userfile},1 | qsub
rppath="~/representation_presenter/representation_presenter.py"
rawdatadir="~/log-data/"
userdatadir="~/user-data/"
writedir="~/results/"
groupby="object_id_path"
vectorsizes="50 100 200 400"
windowsizes="4 8 16"

# Uncomment for command line control
# vectorsize=$1 # e.g. 50
# windowsize=$2 # e.g. 4

#rp_output=rp_output_$(date +%Y-%m-%d-%T).txt
#touch $rp_output
rp_outfile=${writedir}rp_output$(date +%Y-%m-%d-%T).txt
> $rp_outfile
for course in ${rawdatadir}*.tsv
do
    echo $course
    course=${course#*/event-data/}
    output=${course%_parsed*}
    echo $output
    userfile=${userdatadir}${output}results.tsv
    if [ ! -f "$userfile" ]; then
        echo "File does not exist"
    else
        echo "File exists"
        for vectorsize in $vectorsizes
        do
    	    for windowsize in $windowsizes
            do
            	filebase=${output}_${groupby}_${vectorsize}v_${windowsize}w.tsv
    	        echo python3 ${rppath} -i ${rawdatadir}${course} -o ${filebase},4 -d ${writedir} -v ${vectorsize} -w ${windowsize} -t ~/bhtsne/ -g ${groupby} -k userid -s time -f ${userfile},1 | qsub -k oe
    		    echo $filebase >> $rp_outfile
            done
        done
    fi
done
