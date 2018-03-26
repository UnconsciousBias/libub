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
