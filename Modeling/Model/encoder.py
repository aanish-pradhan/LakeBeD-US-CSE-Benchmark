import torch.nn as nn

class Encoder(nn.Module):
	"""
	Defines a unidirectional LSTM-RNN for encoding an input time series sample.
	"""

	def __init__(self, 
				 hidden_state_dim: int,
				 num_input_features: int,
				 dropout_rate: float = 0.0,
				 num_recurrent_layers: int = 1
				 ):
		"""
		Encoder object constructor.
		"""
		
		super(Encoder, self).__init__()
		self.hidden_state_dim = hidden_state_dim
		self.num_input_features = num_input_features
		self.dropout_rate = dropout_rate
		self.num_recurrent_layers = num_recurrent_layers

		# Encoder LSTM-RNN
		self.encoder_lstm = nn.LSTM(input_size = self.num_input_features,
							  		hidden_size = self.hidden_state_dim,
									num_layers = self.num_recurrent_layers,
									bias = True,
									batch_first = True,
									dropout = self.dropout_rate,
									bidirectional = False)

	def forward(self, x):
		"""
		Define forward propagation through the encoder
		"""

		lstm_output, (final_hidden_state, final_cell_state) = self.encoder_lstm(x.view(x.shape[0], x.shape[1], self.num_input_features))
		return lstm_output, (final_hidden_state, final_cell_state)
