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
	
	def __train__epoch__(self, train_loader, optimizer):
		self.train()

		total_train_cost = 0.0
		total_train_batches = 0
		predictions = []

		for batch_idx, batch in enumerate(train_loader):

			# Move each tensor in the batch to DEVICE
			batch = [tensor.to(self.device) for tensor in batch]

			# Extract out inputs and targets
			input_batch, target_batch = batch

			# Forward propagation
			prediction = self.forward(input_batch)
			predictions.append(prediction.detach().cpu().numpy())

			# Check if all targets in samples of the batch are NaN and omit if so omit
			reshaped_targets = target_batch.view(-1, target_batch.size(-1))
			if torch.isnan(reshaped_targets).all():
				continue

			# Compute masked cost
			cost = self.masked_rmse_cost(prediction, target_batch)
			if torch.isnan(cost) or not cost.requires_grad:
				continue

			# Zero gradients in the optimizer
			optimizer.zero_grad()

			# Backpropagation
			cost.backward()
			torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1.0)

			# Update parameters
			optimizer.step()

			total_train_cost += cost.item()
			total_train_batches += 1

		average_train_cost = total_train_cost / total_train_batches if total_train_batches > 0 else float("inf")
		
		return average_train_cost, predictions
	
	def __validate_epoch__(self, validation_loader):
		self.eval()

		with torch.no_grad():
			total_validation_cost = 0.0
			total_validation_batches = 0
			predictions = []

			for batch_idx, batch in enumerate(validation_loader):

				# Move each tensor in the batch to DEVICE
				batch = [tensor.to(self.device) for tensor in batch]

				# Extract out inputs and targets
				input_batch, target_batch = batch

				# Forward propagation
				prediction = self.forward(input_batch)
				predictions.append(prediction.detach().cpu().numpy())

				# Check if all targets in samples of the batch are NaN and omit if so omit
				reshaped_targets = target_batch.view(-1, target_batch.size(-1))
				if torch.isnan(reshaped_targets).all():
					continue

				# Compute masked cost
				cost = self.masked_rmse_cost(prediction, target_batch)
				if torch.isnan(cost):
					continue

				total_validation_cost += cost.item()
				total_validation_batches += 1

			average_validation_cost = total_validation_cost / total_validation_batches if total_validation_batches > 0 else float("inf")

		return average_validation_cost, predictions

	def __test_model__(self, test_loader):
		self.eval()

		with torch.no_grad():
			total_testing_cost = 0.0
			total_testing_batches = 0
			predictions = []

			for batch_idx, batch in enumerate(test_loader):

				# Move each tensor in the batch to DEVICE
				batch = [tensor.to(self.device) for tensor in batch]

				# Extract out inputs and targets
				input_batch, target_batch = batch

				# Forward propagation
				prediction = self.forward(input_batch)
				predictions.append(prediction.detach().cpu().numpy())

				# Check if all targets in samples of the batch are NaN and omit if so omit
				reshaped_targets = target_batch.view(-1, target_batch.size(-1))
				if torch.isnan(reshaped_targets).all():
					continue

				# Compute masked cost
				cost = self.masked_rmse_cost(prediction, target_batch)
				if torch.isnan(cost):
					continue

				total_testing_cost += cost.item()
				total_testing_batches += 1

			average_testing_cost = total_testing_cost / total_testing_batches if total_testing_batches > 0 else float("inf")

		return average_testing_cost, predictions			
			
	def train_model(self, 
					training_loader, 
					validation_loader, 
					test_loader, 
					num_epochs, 
					learning_rate, 
					lr_scheduler_threshold = 1e-4,
					lr_scheduler_patience = 3, 
					weight_decay = 0.0, 
					early_stopping_patience = 5):

		# Optimizer
		optimizer = optim.Adam(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold = lr_scheduler_threshold, patience = lr_scheduler_patience)

		# Model outputs
		final_training_predictions = None
		final_validation_predictions = None
		final_testing_predictions = None
		average_training_costs = []
		average_validation_costs = []
		average_testing_cost = None
		lr_per_epoch = []

		# Early stoppping
		best_validation_cost = float("inf")
		epochs_without_improvement = 0

		for epoch in tqdm(range(num_epochs)):

			lr_per_epoch.append(optimizer.param_groups[0]["lr"])

			# Training
			average_training_cost, training_prediction = self.__train__epoch__(training_loader, optimizer)
			average_training_costs.append(average_training_cost)
			final_training_predictions = training_prediction

			# Validation
			average_validation_cost, validation_prediction = self.__validate_epoch__(validation_loader)
			average_validation_costs.append(average_validation_cost)
			final_validation_predictions = validation_prediction

			# Adjust learning rate
			lr_scheduler.step(average_validation_cost)

			# Early stopping check
			if average_validation_cost < best_validation_cost:
				best_validation_cost = average_validation_cost
				epochs_without_improvement = 0 # Reset counter
			else:
				epochs_without_improvement += 1

			if epochs_without_improvement >= early_stopping_patience:
				print(f"Early stopping at epoch {epoch + 1} with best validation cost: {best_validation_cost:.4f}")
				break

			print(f"EPOCH {epoch + 1}")
			print(f"Average Training Cost (RMSE): {average_training_cost:.4f}\tAverage Validation Cost (RMSE): {average_validation_cost:.4f}")

		# Testing
		average_testing_cost, final_testing_predictions = self.__test_model__(test_loader)
		print(f"Average Testing Cost (RMSE): {average_testing_cost}")

		return final_training_predictions, final_validation_predictions, final_testing_predictions, average_training_costs, average_validation_costs, average_testing_cost, lr_per_epoch

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
