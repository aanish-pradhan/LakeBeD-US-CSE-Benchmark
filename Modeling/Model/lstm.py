from .decoder import Decoder
from .encoder import Encoder
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class LSTM(nn.Module):
	"""
	Defines a seq2seq LSTM-RNN for multivariate-to-multivariate time series 
	prediction.
	"""

	def __init__(self,
			  	 device,
				 hidden_state_dim: int,
				 horizon_window: int,
				 num_input_features: int,
				 num_target_features: int,
				 decoder_dropout_rate: float = 0.0,
				 decoder_num_recurrent_layers: int = 1,
				 encoder_dropout_rate: float = 0.0,
				 encoder_num_recurrent_layers: int = 1) -> nn.Module:

		super(LSTM, self).__init__()
		self.device = device
		self.hidden_state_dim = hidden_state_dim
		self.horizon_window = horizon_window
		self.num_input_features = num_input_features
		self.num_target_features = num_target_features
		self.decoder_dropout_rate = decoder_dropout_rate
		self.decoder_num_recurrent_layers = decoder_num_recurrent_layers
		self.encoder_dropout_rate = encoder_dropout_rate
		self.encoder_num_recurrent_layers = encoder_num_recurrent_layers
		
		# Encoder LSTM-RNN
		self.encoder = Encoder(hidden_state_dim = self.hidden_state_dim,
						 	   num_input_features = self.num_input_features,
							   dropout_rate = self.encoder_dropout_rate,
							   num_recurrent_layers = self.encoder_num_recurrent_layers).to(self.device)
		
		# Decoder LSTM-RNN
		self.decoder = Decoder(hidden_state_dim = self.hidden_state_dim,
							   num_input_features = self.num_input_features,
							   num_target_features = self.num_target_features,
							   dropout_rate = self.decoder_dropout_rate,
							   num_recurrent_layers = self.decoder_num_recurrent_layers).to(self.device)
		
		# Move LSTM to DEVICE
		self.to(self.device)
		
	def forward(self, x):
		"""
		Defines forward propagation through the autoencoder
		"""

		# Encode the input sequence
		encoder_output, encoder_final_hidden_state = self.encoder(x)

		# Initialize the decoder hidden state with the encoder's final hidden state
		decoder_hidden = encoder_final_hidden_state

		# Prepare outputs tensor
		batch_size = x.size(0)
		outputs = torch.zeros(batch_size, self.horizon_window, self.num_target_features, device = self.device)

		# Autoregressive decoding
		decoder_input = torch.zeros(batch_size, 1, self.num_target_features, device=self.device)  # Shape: (batch_size, 1, output_size)

		for t in range(self.horizon_window):
			# Pass through the decoder
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
			#print("Decoder output", decoder_output)

			# Store the output
			outputs[:, t, :] = decoder_output.squeeze(1)
			# Use the current output as the next input
			decoder_input = decoder_output.detach()  # Detach to prevent backprop through the entire sequence

		return outputs

	def train_model(self, train_loader, validation_loader, num_epochs, learning_rate):
		"""
		Trains the LSTM model using a masked RMSE cost to handle missing targets.

		Args:
			train_loader (DataLoader): DataLoader for training data.
			val_loader (DataLoader): DataLoader for validation data.
			num_epochs (int): Number of training epochs.

		Returns:
			None
		"""
		
		optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay = )
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold = 1e-2)

		training_predictions = []
		validation_predictions = []
		average_training_costs = []
		average_validation_costs = []

		for epoch in tqdm(range(num_epochs)):
			self.train()
			total_train_cost = 0.0
			total_train_batches = 0

			for batch_idx, batch in enumerate(train_loader):	

				# Move each tensor in the batch to DEVICE
				batch = [tensor.to(self.device) for tensor in batch]

				input_batch, target_batch = batch

				# Forward propagation
				outputs = self.forward(input_batch)

				if epoch == num_epochs - 1:
					training_predictions.append(outputs.detach().cpu().numpy())

				# Check if all targets in samples of the batch are NaN and omit if so
				reshaped_targets = target_batch.view(-1, target_batch.size(-1))
				if torch.isnan(reshaped_targets).all():
					#print("Skipping batch with all NaN targets.")
					continue

				# Compute masked cost
				cost = self.masked_rmse_cost(outputs, target_batch)
				if torch.isnan(cost) or not cost.requires_grad:
					#print("Skipping batch due to NaN loss.")
					continue

				# Zero gradients in the optimizer
				optimizer.zero_grad()

				# Backpropagation
				cost.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

				# Update parameters
				optimizer.step()

				total_train_cost += cost.item()
				total_train_batches += 1

			average_train_cost = total_train_cost / total_train_batches if total_train_batches > 0 else float('inf')
			average_training_costs.append(average_train_cost)
			
			# Validation phase
			self.eval()
			total_val_cost = 0.0
			total_val_batches = 0

			with torch.no_grad():
				for batch_idx, batch in enumerate(validation_loader):

					# Move each tensor in the batch to DEVICE
					batch = [tensor.to(self.device) for tensor in batch]

					input_batch, target_batch = batch

					# Check if all targets in samples of the batch are NaN and omit if so
					reshaped_targets = target_batch.view(-1, target_batch.size(-1))
					if torch.isnan(reshaped_targets).all():
						continue

					# Forward propagation
					outputs = self.forward(input_batch)

					if epoch == num_epochs - 1:
						validation_predictions.append(outputs.detach().cpu().numpy())

					# Compute masked cost
					cost = self.masked_rmse_cost(outputs, target_batch)
					if torch.isnan(cost):
						continue

					total_val_cost += cost.item()
					total_val_batches += 1

			average_validation_cost = total_val_cost / total_val_batches if total_val_batches > 0 else float('inf')
			average_validation_costs.append(average_validation_cost)

			# Adjust the learning rate
			lr_scheduler.step(average_validation_cost)

			print(f"EPOCH {epoch + 1}")
			print(f"Average Training Cost (RMSE): {average_train_cost:.4f}\tAverage Validation Cost (RMSE): {average_validation_cost:.4f}")

		return training_predictions, validation_predictions, average_training_costs, average_validation_costs

	def masked_rmse_cost(self, outputs, targets):
		mask = ~torch.isnan(targets)
		
		# Ensure there are valid targets
		if not mask.any():
			raise ValueError("All target values are NaN. Cannot compute MSE.")

		# Compute MSE only on valid targets
		mse = F.mse_loss(outputs[mask], targets[mask], reduction = 'mean')

		# Add epsilon to prevent sqrt(0)
		rmse = torch.sqrt(mse)

		return rmse
