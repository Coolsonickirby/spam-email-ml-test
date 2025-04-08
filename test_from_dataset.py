import torch
from torch.autograd import Variable
from model import *
from tokenize_dataset import tokenize, words_table
import json
import random

class Predictor(ModelWrapper):
    def __init__(self, saved_dir='./', file_name = 'Model190.pth'):
        vocab_size = len(words_table)
        self.set_params(vocab_size, train_on_gpu = False)
        self.set_model(do_print = False)
        self.load_state_dict(saved_dir, file_name, do_print = False)

    def predict(self, text, unnecessary = ["-", ".", ",", "/", ":", "@", "'", "!"]):	
        text = text.lower()
        text = ''.join([c for c in text if c not in unnecessary])
        text = [text]
        vectorized_seqs = [tokenize(text[0])]

        seq_tensor_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_tensor_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_tensor_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        self.model.eval()
        output = self.model(seq_tensor, seq_tensor_lengths)
        return output.item()
    
p = Predictor(file_name="Model190.pth")

source_dataset = "dataset.json"
dataset = {}
with open(source_dataset, "r") as f:
    dataset = json.load(f)


test_data = []
set_len = len(dataset["ham"]) - 1

total_per_set = 300

for x in range(total_per_set):
    while dataset["ham"][set_len - x] == "":
        set_len -= 1
    test_data.append({
        "type": "ham",
        "msg": dataset["ham"][set_len - x],
        "prob": 0.0
    })

set_len = len(dataset["spam"]) - 1
for x in range(total_per_set):
    while dataset["spam"][set_len - x] == "":
        set_len -= 1
    test_data.append({
        "type": "spam",
        "msg": dataset["spam"][set_len - x],
        "prob": 0.0
    })

random.shuffle(test_data)

correct_spam = 0
correct_ham = 0

for x in range(len(test_data)):
    test_data[x]["prob"] = p.predict(test_data[x]["msg"])
    if test_data[x]["prob"] >= 0.7 and test_data[x]["type"] == "ham":
        correct_ham += 1
    elif test_data[x]["prob"] < 0.7 and test_data[x]["type"] == "spam":
        correct_spam += 1


print("Correct Ham: %d out of %d (%f)" % (correct_ham, total_per_set, (float(correct_ham) / float(total_per_set) * 100)))
print("Correct Spam: %d out of %d (%f)" % (correct_spam, total_per_set, (float(correct_spam) / float(total_per_set) * 100)))




with open("outcome_collected.json", "w+") as f:
    json.dump(test_data, f, indent=4)