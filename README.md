# Semantic Model

This tool supplements CAHLR's Representation Presenter and d3-scatterplot tools. Given the Word2vec Vectorfile (full vector space) and 2D Vectorfile with raw features, output types 2 and 3 from calling representation_presenter.py, and the text-valued column from the 2D Vectorfile you would like to perform semantic analysis on, semantic_model.py will output the files d3-scatterplot needs for semantic models.

In particular, semantic_model.py will create bag of words representations for the text-valued column, and train the coefficients of the full vector space to predict those bag of words.
Then, it will also cluster points based on 1) the true bag of words and 2) the bag of word predictions using the logistic regression parameters trained in the first step. It will evaluate both clusterings using the cosine proximity of the points in 1) the full vector space and 2) the 2D vector space.

It will save the bag of words representations and two cluster assignments into the 2D Vectorfile, along with the vocabulary, weights and biases from the learned model, and the cluster scores.

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
    -k num_clusters
    -b tf_bias
    -e num_epochs
    -i use_idf
Where `write_directory` is where you would like to save the output files to, `scorefile_location` is where you would like the clustering scores to be written to, `num_clusters` is the number of clusters you would like to classify your data into, `-b` is the term-frequency bias (not currently in use), `-e` is the number of epochs to train logistic regression for, and `use_idf` is whether or not to use inverse document frequency in tf-TfidfVectorizer.

## Example:
With the included vectorfile and rawfile in the current directory,

	python3 semantic_model.py -v example_vector.tsv -r example_raw.tsv -t textcol
