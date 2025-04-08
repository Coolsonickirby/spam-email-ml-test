import numpy as np
from torch import device
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

DEBUG = False

class ModelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,\
                 drop_out_in_lstm, drop_out, output_size, device):

        super().__init__()
        self.device = device
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_out_in_lstm, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_out)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, seq_lengths):

        # embeddings
        embedded_seq_tensor = self.embedding(x)
        if DEBUG:
          print("embedded_seq_tensor = self.embedding(x)", embedded_seq_tensor.shape)
                
        # pack, remove pads
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        if DEBUG:
          print("packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)")
          print(packed_input.data.shape)
          print(packed_input.batch_sizes.shape)
        
        # lstm
        packed_output, (ht, ct) = self.lstm(packed_input, None)
        if DEBUG:
          print("packed_output, (ht, ct) = self.lstm(packed_input, None)")
          print(packed_output.data.shape)
          print(packed_output.batch_sizes.shape)
          print("ht")
          print(ht.shape)
        
        # unpack, recover padded sequence
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # output : batch_size X max_seq_len X hidden_dim
        if DEBUG:
          print("output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)")
          print(output.shape)
          print(input_sizes)
       
        # gather the last output in each batch
        last_idxs = (input_sizes - 1).to(self.device) # last_idxs = input_sizes - torch.ones_like(input_sizes)
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() # [batch_size, hidden_dim]
        if DEBUG:
          print(output.shape) 
        
        # dropout and fully-connected layer
        output = self.dropout(output)
        output = self.fc(output).squeeze()
        if DEBUG:
          print("output = self.fc(output)", output.shape)
               
        # sigmoid function
        output = self.sig(output)
        
        return output


class ModelWrapper(object):
  
	def set_params(self, vocab_size, \
					   embedding_dim = 100, \
					   hidden_dim = 15, \
					   n_layers = 2, \
					   drop_out_in_lstm = 0.2, \
					   drop_out = 0.2, \
					   output_size = 1, \
					   train_on_gpu = True):
    
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.n_layers = 2
		self.drop_out_in_lstm = drop_out_in_lstm
		self.drop_out = drop_out
		self.output_size = output_size
		self.train_on_gpu = train_on_gpu
		self.device = "cuda" if torch.cuda.is_available() and train_on_gpu else "cpu" 

	def set_model(self, do_print = True):
		self.model = ModelLSTM(self.vocab_size, self.embedding_dim, self.hidden_dim, self.n_layers, \
					 self.drop_out_in_lstm, self.drop_out, self.output_size, self.device)
		self.model = self.model.to(self.device)
		if do_print:
			print(self.model)

	def train(self, train_loader, valid_loader, criterion = "default", optimizer="default", learning_rate = 0.03, use_scheduler = True, \
         epochs = 6, validate_every = 10, gradient_clip = 5):

		if criterion == "default" :
			criterion = nn.BCELoss()
		print(criterion)
     

		if optimizer == "default" :
			optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		print(optimizer)

		if use_scheduler :
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 2)

		counter = 0

		self.model.train()
	
		val_losses = []
		val_min_loss = 1000000

		for e in range(epochs):

			if use_scheduler :
				scheduler.step(e)

			for seq_tensor, seq_tensor_lengths, label in iter(train_loader):
				counter += 1

				seq_tensor = seq_tensor.to(self.device)
				seq_tensor_lengths = seq_tensor_lengths.to(self.device)
				label = label.to(self.device)

				# get the output from the model
				output = self.model(seq_tensor, seq_tensor_lengths)

				# calculate the loss and perform backprop
				loss = criterion(output, label.float())
				optimizer.zero_grad() 
				loss.backward()

				# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
				nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
				optimizer.step()

				# loss stats
				if counter % validate_every == 0:
					# Get validation loss
					val_losses_in_itr = []
					sums = []
					sizes = []

					self.model.eval()

					for seq_tensor, seq_tensor_lengths, label in iter(valid_loader):

						seq_tensor = seq_tensor.to(self.device)
						seq_tensor_lengths = seq_tensor_lengths.to(self.device)
						label = label.to(self.device)
						output = self.model(seq_tensor, seq_tensor_lengths)

						# losses
						val_loss = criterion(output, label.float())     
						val_losses_in_itr.append(val_loss.item())

						# accuracy
						binary_output = (output >= 0.5).short() # short(): torch.int16
						right_or_not = torch.eq(binary_output, label)
						sums.append(torch.sum(right_or_not).float().item())
						sizes.append(right_or_not.shape[0])

					val_losses.append(np.mean(val_losses_in_itr))
					if val_min_loss > val_losses[-1]:
						val_min_loss = val_losses[-1]
						self.save_state_dict('./', 'Model{}.pth'.format(counter))

					accuracy = np.sum(sums) / np.sum(sizes)

					self.model.train()
					print("Epoch: {:2d}/{:2d}\t".format(e+1, epochs),
						  "Steps: {:3d}\t".format(counter),
						  "Loss: {:.5f}\t".format(loss.item()),
						  "Val Loss: {:.5f}\t".format(np.mean(val_losses_in_itr)),
						  "Accuracy: {:.3f}".format(accuracy))    

	def test(self, test_loader, criterion = "default"):
		if criterion == "default":
			criterion = nn.BCELoss()

		test_losses = []
		sums = []
		sizes = []

		self.model.eval()

		test_losses = []
		
		for seq_tensor, seq_tensor_lengths, label in iter(test_loader):
			seq_tensor = seq_tensor.to(self.device)
			seq_tensor_lengths = seq_tensor_lengths.to(self.device)
			label = label.to(self.device)
			output = self.model(seq_tensor, seq_tensor_lengths)

			# losses
			test_loss = criterion(output, label.float())     
			test_losses.append(test_loss.item())

			# accuracy
			binary_output = (output >= 0.5).short() # short(): torch.int16
			right_or_not = torch.eq(binary_output, label)
			sums.append(torch.sum(right_or_not).float().item())
			sizes.append(right_or_not.shape[0])

		accuracy = np.sum(sums) / np.sum(sizes)
		print("Test Loss: {:.6f}\t".format(np.mean(test_losses)),
		"Accuracy: {:.3f}".format(accuracy))
    
	def load_state_dict(self, saved_dir='./', file_name='saved_model.pth', do_print = True):
		output_path = os.path.join(saved_dir, file_name)
		checkpoint = torch.load(output_path, map_location=self.device)
		state_dict = checkpoint['net']
		self.model.load_state_dict(state_dict)
		if do_print:
			for name, param in self.model.named_parameters():
				if param.requires_grad:
					print(name, param.data.shape)

	def save_state_dict(self, saved_dir='./', file_name='saved_model.pth', do_print = True):
		os.makedirs(saved_dir, exist_ok=True)
		check_point = {
			'net': self.model.state_dict()
		}
		output_path = os.path.join(saved_dir, file_name)
		torch.save(check_point, output_path)
		if do_print:
			print("saved as", output_path)