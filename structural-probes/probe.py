"""Classes for specifying probe pytorch modules."""

import torch.nn as nn
import torch
# import memory

class Probe(nn.Module):
  pass

class TwoWordPSDProbe(Probe):
  """ Computes squared L2 distance after projection by a matrix.

  For a batch of sentences, computes all n^2 pairs of distances
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('Constructing TwoWordPSDProbe')
    super(TwoWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])

  def forward(self, batch):
      """Compute distances in chunks to minimize memory usage."""
      cpu_batch = batch.cpu()
      cpu_proj = self.proj.cpu()
      transformed = torch.matmul(batch, self.proj)
      batchlen, seqlen, rank = transformed.size()
      
      # Pre-allocate the output tensor
      squared_distances = torch.zeros(batchlen, seqlen, seqlen, device=transformed.device)
      
      # Process in chunks to reduce memory footprint
      chunk_size = min(32, seqlen)  # Adjust based on your memory constraints
      
      for i in range(0, seqlen, chunk_size):
          end_i = min(i + chunk_size, seqlen)
          chunk_i = transformed[:, i:end_i, :]
          
          for j in range(0, seqlen, chunk_size):
              end_j = min(j + chunk_size, seqlen)
              chunk_j = transformed[:, j:end_j, :]
              
              # Compute pairwise distances between chunks
              chunk_i_expanded = chunk_i.unsqueeze(2).expand(-1, -1, end_j-j, -1)
              chunk_j_expanded = chunk_j.unsqueeze(1).expand(-1, end_i-i, -1, -1)
              diffs = chunk_i_expanded - chunk_j_expanded
              squared_diffs = diffs.pow(2)
              squared_distances[:, i:end_i, j:end_j] = torch.sum(squared_diffs, -1)
      
      return squared_distances.to(batch.device)



class OneWordPSDProbe(Probe):
  """ Computes squared L2 norm of words after projection by a matrix."""

  def __init__(self, args):
    print('Constructing OneWordPSDProbe')
    super(OneWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (Bh_i)^T(Bh_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = transformed.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        transformed.view(batchlen* seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class OneWordNonPSDProbe(Probe):
  """Computes a bilinear affinity between each word representation and itself.
  
  This is different from the probes in A Structural Probe... as the
  matrix in the quadratic form is not guaranteed positive semi-definite
  
  """

  def __init__(self, args):
    print('Constructing OneWordNonPSDProbe')
    super(OneWordNonPSDProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (h_i^T)A(h_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = batch.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        batch.view(batchlen*seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class TwoWordNonPSDProbe(Probe):
  """ Computes a bilinear function of difference vectors.

  For a batch of sentences, computes all n^2 pairs of scores
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('TwoWordNonPSDProbe')
    super(TwoWordNonPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n^2 pairs of difference scores 
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (h_i-h_j)^TA(h_i-h_j) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    """
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    diffs = (batch_square - batch_square.transpose(1,2)).view(batchlen*seqlen*seqlen, rank)
    psd_transformed = torch.matmul(diffs, self.proj).view(batchlen*seqlen*seqlen,1,rank)
    dists = torch.bmm(psd_transformed, diffs.view(batchlen*seqlen*seqlen, rank, 1))
    dists = dists.view(batchlen, seqlen, seqlen)
    return dists
