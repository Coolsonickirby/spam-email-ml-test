import torch
from torch.autograd import Variable
from model import *
from tokenize_dataset import tokenize, words_table

class Predictor(Model_wrapper):
    def __init__(self, saved_dir='./', file_name = 'lstm_model_saved_at_90.pth'):
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
    
p = Predictor(file_name="lstm_model_saved_at_90.pth")

# The higher the number, the more likely it's ham and not spam
while True:
    res = input("Enter a sentence: ").strip()
    if res == "":
        exit()
    print(p.predict(res))