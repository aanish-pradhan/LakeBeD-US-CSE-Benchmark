import torch.nn as nn

class Decoder(nn.Module):
	"""
	Defines a unidirectional LSTM-RNN for decoding and generating an output 
	time series prediction.
	"""

	def __init__(self,
			  	 hidden_state_dim: int,
				 num_input_features: int,
				 num_target_features: int,
				 dropout_rate: float = 0.0,
				 num_recurrent_layers: int = 1) -> nn.Module:
		"""
		Decoder object constructor.
		"""
		
		super(Decoder, self).__init__()
		self.hidden_state_dim = hidden_state_dim
		self.num_input_features = num_input_features
		self.num_target_features = num_target_features
		self.dropout_rate = dropout_rate
		self.num_recurrent_layers = num_recurrent_layers

		# Decoder LSTM-RNN
		self.decoder_lstm = nn.LSTM(input_size = self.num_target_features,
							  		hidden_size = self.hidden_state_dim,
									num_layers = self.num_recurrent_layers,
									bias = True,
									batch_first = True,
									dropout = self.dropout_rate,
									bidirectional = False)
		self.linear = nn.Linear(self.hidden_state_dim, self.num_target_features)

	def forward(self, decoder_input, hidden_state):
		"""
		Define forward propagation through the decoder
		"""

		lstm_output, (final_hidden_state, final_cell_state) = self.decoder_lstm(decoder_input, hidden_state)
		output = self.linear(lstm_output)

		return output, (final_hidden_state, final_cell_state)