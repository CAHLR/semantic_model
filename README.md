# Semantic Model

This tool supplements CAHLR's Representation Presenter and d3-scatterplot tools. When given:

    - A text-valued column from the 2D Vectorfile you would like to perform semantic analysis on
    - A Word2vec Vectorfile (full vector space, output type 2 from `representation_presenter.py`)
    - A 2D Vectorfile with raw features (output type 2 and 3 from `representation_presenter.py`)


`semantic_model.py` will output the files d3-scatterplot needs to visualize semantic models.

In particular, `semantic_model.py` will create bag of words representations for the text-valued column, and train a logistic regression model using the full vector space to predict those bag of words. Note: we do not train with the empty bag of words.

For multiple-choice fields, the vocabulary of the bag-of-words will be small, and the bag-of-words vector will be one-hot if it select 1, or multi-hot if it was select several. The only condition is that the value in these columns need to be like "A" "B" "C" "D" for one-hot, and "A B" "A B C" "C D" etc. for multi-hot.

Then, it will also cluster vectors and score the clustering based on 1) the true bag of words--only vectors with nonempty bags--and 2) the bag of word predictions--for all vectors--using the logistic regression coefficients trained in the first step. It will evaluate both clusterings using the cosine proximity of the points in 1) the full vector space and 2) the 2D vector space using silhouette score.

It will
    - save the bag of words representations of the vectors in a new file,
    - append the predictions and the two cluster assignments into the original 2D Vectorfile,
    - save each in their own file: the vocabulary, weights and biases from the learned model,
    - save the clustering scores will be saved in the scorefile.

## Synopsis

    python3 semantic_model.py [options]

## Options
    -v vectorfile_path
    -r rawfile_path
    -t textcolumn
Where  `vectorfile` and `rawfile` are outputs from `representation_presenter.py`, and `blobcolumn` is the name of the text feature within `rawfile` to perform semantic analysis on.

### Optional options
    -d write_directory (if not current directory)
    -s scorefile_location (if not ./scorefile.txt)
    -f finishedfile_location (if not ./finishedfile.txt)
    -k num_clusters
    -b tf_bias
    -e num_epochs
    -i use_idf
    -p predict
    -n cluster_input
    -l cluster_eval
Where `write_directory` is where you would like to save the output files to, `scorefile_location` is where you would like the clustering scores to be written to, `finishedfile_location` is where you would like the completion of a run to be recorded, `num_clusters` is the number of clusters you would like to classify your data into, `tf_bias` is the term-frequency bias, `num_epochs` is the number of epochs to train logistic regression for, `use_idf` is whether or not to use inverse document frequency in tf-TfidfVectorizer, and 'predict' is whether or not to extrapolate the motivations of the no-response entries.


`cluster_input` is which word representation input to use for clustering: "bow" for the actual bag of words of the point or "softmax" for the logistic regression model's prediction of words for the point, any other string will output scores corresponding to both inputs. `cluster_eval` is which vector space to calculate the silhouette score of the clusterings: "vector" for full vector space or "2d" for the tSNE reduced vector space, any other string will evaluate to outputting scores corresponding to both vector spaces.

## Example:
With the included vectorfile and rawfile in the current directory,

	python3 semantic_model.py -v example_vector.tsv -r example_raw.tsv -t textcol


Example bash scripts are included for guidance on mass use of this python script.
