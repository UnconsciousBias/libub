from sklearn.feature_extraction.text import TfidfVectorizer

class AutoEncoderMixin(object):
    """ Mixin class for all sklearn-like Autoencoders """

    def reconstruct(self, X, y=None):
        """ Transform data, then inverse transform it """
        hidden = self.transform(X)
        return self.inverse_transform(hidden)


class EmbeddedVectorizer(TfidfVectorizer):

    """ Weighted Bag-of-embedded-Words"""

    def __init__(self, embedding, index2word, **tfidf_params):
        """
        Arguments
        ---------

        embedding: V x D embedding matrix
        index2word: list of words with indices matching V
        """
        super(EmbeddedVectorizer, self).__init__(self, vocabulary=index2word,
                                                 **tfidf_params)
        self.embedding = embedding

    def fit(self, raw_documents, y=None):
        super(EmbeddedVectorizer, self).fit(raw_documents)
        return self

    def transform(self, raw_documents, __y=None):
        sparse_scores = super(EmbeddedVectorizer,
                              self).transform(raw_documents)
        # Xt is sparse counts
        return sparse_scores @ self.embedding

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents, y)


class GensimEmbeddedVectorizer(EmbeddedVectorizer):
    """
    Shorthand to create an embedded vectorizer using a gensim KeyedVectors
    object, such that the vocabularies match.
    """

    def __init__(self, gensim_vectors, **tfidf_params):
        """
        Arguments
        ---------
        `gensim_vectors` is expected to have index2word and syn0 defined
        """
        embedding = gensim_vectors.syn0
        index2word = gensim_vectors.index2word
        super(GensimEmbeddedVectorizer, self).__init__(embedding,
                                                       index2word,
                                                       **tfidf_params)


class PaddedSequence(object):  # should subclass sklearn transformer
    """ A Vectorizer that transforms text into padded sequences instead of bag
    of words / ngrams. Drop-in replacement for CountVectorizer /
    TfIdfVectorizer. CountVectorizer is used to build the vocabulary. """
    unk_ = 1
    pad_ = 0
    offset_ = 2

    def __init__(self, sort=False, fix_empty=True, drop_unk=False, **cv_params):
        """
        unk_token: If None, unknown tokens are dropped. Else the specified
        index will be used for unk tokens.

        Other keyword arguments are directly passed to CountVectorizer
        """
        super(PaddedSequence, self).__init__()
        assert 'ngram_range' not in cv_params, "This does not make sense."
        self.cv = CountVectorizer(**cv_params)
        self.fix_empty = fix_empty
        self.sort = sort
        self.drop_unk = drop_unk
        self.vocabulary_size_ = None

    def fit(self, raw_documents):
        """ Constructs vocabulary from raw documents """
        self.cv.fit(raw_documents)
        # store vocabulary size
        self.vocabulary_size_ = len(self.cv.vocabulary_) + self.offset_
        return self

    def transform(self, raw_documents, return_lengths=False):
        """ Transforms a batch of raw_documents and pads to max length within batch """

        vocab = self.cv.vocabulary_

        # Tokenize sentences
        analyze = self.cv.build_analyzer()
        sentences = (analyze(doc) for doc in raw_documents)  # generator

        if self.drop_unk:
            sentences = [[vocab[w] + self.offset_ for w in s if w in vocab] for s in sentences]
        else:
            sentences = [[vocab[w] + self.offset_ if w in vocab else self.unk_
                          for w in s] for s in sentences]

        if self.fix_empty:
            # Place a single unk token in empty sentences
            sentences = [s if s else [self.unk_] for s in sentences]

        if self.sort:
            sentences.sort(key=len, reverse=True)
            max_length = len(sentences[0])
        else:
            max_length = max(map(len, sentences))

        n_samples = len(sentences)
        padded_sequences = np.empty((n_samples, max_length), dtype='int64')
        padded_sequences.fill(self.pad_)
        for i, sentence in enumerate(sentences):
            for j, token in enumerate(sentence):
                padded_sequences[i, j] = token

        if return_lengths:
            lengths = list(map(len, sentences))
            return padded_sequences, lengths
        else:
            return padded_sequences

    def fit_transform(self, raw_documents, **transform_params):
        """ Applies fit, then transform on raw documents """
        return self.fit(raw_documents).transform(raw_documents, **transform_params)

    def inverse_transform(self, sequences, join=None):
        """ Inverse transforms an iterable of iterables holding indices """
        reverse_vocab = {idx + self.offset_: word for word, idx in self.cv.vocabulary_.items()}
        assert self.pad_ not in reverse_vocab
        assert self.unk_ not in reverse_vocab
        reverse_vocab[self.pad_] = '<PAD>'
        reverse_vocab[self.unk_] = '<UNK>'
        sentences = [[reverse_vocab[t] for t in s] for s in sequences]
        if join is None:
            return sentences
        else:
            join_str = str(join)
            return [join_str.join(s) for s in sentences]
