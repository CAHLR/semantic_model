# Semantic Model

This tool supplements CAHLR's Representation Presenter and d3-scatterplot tools. Given the Word2vec Vectorfile and 2D Vectorfile with raw features (output types 2 and 3 from calling representation_presenter.py) and the text-valued column you would like to perform semantic analysis on, semantic_model.py will output the files d3-scatterplot needs for semantic models.

## Synopsis

    python3 semantic_model.py [options]

## Options
    -v vectorfile_path
    -r rawfile_path
    -b blobcolumn_name
Where  `vectorfile` and `rawfile` are outputs from `representation_presenter.py`, and `blobcolumn` is the name of the text feature within `rawfile` to perform semantic analysis on.

## Example:
With the included vectorfile and rawfile in the current directory,

	python3 semantic_model.py -v example_vector.tsv -r example_raw.tsv -b textcol
