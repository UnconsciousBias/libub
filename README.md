# libub

General-purpose extensions for sklearn, gensim, and pytorch.


## sklearn

### GensimEmbeddedVectorizer

The EmbeddedVectorizer combines TfidfVectorizer with a gensim.KeyedVectors object, such that the vocabularies match.


## Gensim 

### peek_word2vec_format

This function peeks at a word2vec file and returns the expected dimensions. **Usage:** `peek_word2vec_format(PATH, binary=True)`

## Torch

### InnerProduct

This `InnerProduct` module computes the inner product of two batches of row vectors.
