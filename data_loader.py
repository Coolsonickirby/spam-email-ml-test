# Sourced from: https://gist.github.com/sijoonlee/0e13ffe5888ad50469c5ab17a5b336ea

import torch.utils.data.sampler as data_sampler

class DataLoader(object):
  def __init__(self, seq_tensor, seq_lengths, label_tensor, batch_size):
    self.batch_size = batch_size
    self.seq_tensor = seq_tensor
    self.seq_lengths = seq_lengths
    self.label_tensor = label_tensor
    self.sampler = data_sampler.BatchSampler(data_sampler.RandomSampler(self.label_tensor), self.batch_size, False)
    self.sampler_iter = iter(self.sampler)
    
  def __iter__(self):
    self.sampler_iter = iter(self.sampler) # reset sampler iterator
    return self

  def _next_index(self):
    return next(self.sampler_iter) # may raise StopIteration

  def __next__(self):
    index = self._next_index()

    subset_seq_tensor = self.seq_tensor[index]
    subset_seq_lengths = self.seq_lengths[index]
    subset_label_tensor = self.label_tensor[index]

    subset_seq_lengths, perm_idx = subset_seq_lengths.sort(0, descending=True)
    subset_seq_tensor = subset_seq_tensor[perm_idx]
    subset_label_tensor = subset_label_tensor[perm_idx]

    return subset_seq_tensor, subset_seq_lengths, subset_label_tensor

  def __len__(self):
    return len(self.sampler)