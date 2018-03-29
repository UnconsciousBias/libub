""" Module for general purpose functions for gensim """


def peek_word2vec_format(path, binary=False):
    """
    Function to peek at the first line of a serialized embedding in
    word2vec format

    Arguments
    ---------
    path: The path to the file to peek
    binary: Whether the file is gzipped

    Returns
    -------
    Tuple of ints split by white space in the first line,
    i.e., for word2vec format the dimensions of the embedding.
    """
    if binary:
        import gzip
        with gzip.open(path, 'r') as peek:
            return tuple(map(int, next(peek).strip().split()))
    else:
        with open(path, 'r') as peek:
            return tuple(map(int, next(peek).strip().split()))
