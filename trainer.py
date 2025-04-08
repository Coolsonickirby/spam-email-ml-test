import torch
from torch.autograd import Variable
import json
from data_loader import DataLoader
from model import *

# Tokenize & Vectorize sequences
vectorized_seqs = []
with open("tokenized_dataset.json", "r") as f:
    data = json.load(f)
    vectorized_seqs = data["spam"][:16000]
    vectorized_seqs.extend(data["ham"][:16000])
    label_tensor = torch.as_tensor([0 for _ in range(16000)] + [1 for _ in range(16000)], dtype = torch.int16)

vocab = []
with open("words_table.json", "r") as f:
    vocab = json.load(f)

# Save the lengths of sequences
seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

# Add padding(0)
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

shuffled_idx = torch.randperm(label_tensor.shape[0])

seq_tensor = seq_tensor[shuffled_idx]
seq_lenghts = seq_lengths[shuffled_idx]
label = label_tensor[shuffled_idx]

PCT_TRAIN = 0.7
PCT_VALID = 0.2

length = len(label)
train_seq_tensor = seq_tensor[:int(length*PCT_TRAIN)] 
train_seq_lengths = seq_lengths[:int(length*PCT_TRAIN)]
train_label = label[:int(length*PCT_TRAIN)]

valid_seq_tensor = seq_tensor[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))] 
valid_seq_lengths = seq_lengths[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))] 
valid_label = label[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]

test_seq_tensor = seq_tensor[int(length*(PCT_TRAIN+PCT_VALID)):]
test_seq_lengths = seq_lengths[int(length*(PCT_TRAIN+PCT_VALID)):]
test_label = label[int(length*(PCT_TRAIN+PCT_VALID)):]

print(train_seq_tensor.shape)
print(valid_seq_tensor.shape)
print(test_seq_tensor.shape)

batch_size = 700
train_loader = DataLoader(train_seq_tensor, train_seq_lengths, train_label, batch_size)
valid_loader = DataLoader(valid_seq_tensor, valid_seq_lengths, valid_label, batch_size)
test_loader = DataLoader(test_seq_tensor, test_seq_lengths, test_label, batch_size)

model = ModelWrapper()
model.set_params(len(vocab))
model.set_model()
model.train(train_loader, valid_loader)