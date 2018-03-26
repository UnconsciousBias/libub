import pytest

def test_imports():
    from libub import peek_word2vec_format
    from libub import GensimEmbeddedVectorizer, EmbeddedVectorizer
    from libub import InnerProduct

from libub import peek_word2vec_format
from libub import GensimEmbeddedVectorizer
from libub import InnerProduct


def test_peek_word2vec_format():
    dims = peek_word2vec_format('test_data/example_embedding.txt', False)
    assert dims[0] == 10000
    assert dims[1] == 10


# TODO test the other modules
