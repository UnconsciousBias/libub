""" Module for general purpose functions for gensim """

import numpy as np
import torch

class InnerProduct(torch.nn.Module):
    """ Computes row-wise dot product of two batches of vectors
    >>> a = torch.autograd.Variable(torch.FloatTensor(np.arange(6).reshape(2,3)))
    >>> b = torch.autograd.Variable(torch.FloatTensor(np.arange(6).reshape(2,3)))
    >>> InnerProduct(1)(a, b).data
    <BLANKLINE>
      5
     50
    [torch.FloatTensor of size 2]
    <BLANKLINE>
    """
    def __init__(self, dim=1):
        """ Another dim than 1 rarely makes sense..."""
        super(InnerProduct, self).__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.bmm(a.unsqueeze(self.dim), b.unsqueeze(self.dim+1)).squeeze()
