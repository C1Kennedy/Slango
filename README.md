# Slango
A trivia game all about learning how to correctly use slang.

Dataset of UrbanDictionary Words:
https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset

Necessary files for running the program:
https://drive.google.com/file/d/1uArnRNAyvOamSQpl7e6d1F9fBa-qvFuH/view

# Installation
- Download corpus from kaggle, download necessary zip folder from google drive link.
- If google drive link is nonfunctional, download and run slango_modelGen.py, this will take a while.
- Download slango_model.py and slango_interface.py.
- Ensure all downloaded and extracted files are within same folder.
- Run slango_interface wherever you run python.

# Errors
- slango_model.main(), when executed, may request "slango_word2vec.model.wv.vectors.npy" or "slango_word2vec.model.syn1neg.npy".
  Both files are in the google drive link.
  Files are too large to place here but are necessary for program.
  One person's computer generated these files when slango_modelGen.py finished executing, but a different computer executing that file did not generate those files.
  Reason or criteria for generation of these files are unknown.
