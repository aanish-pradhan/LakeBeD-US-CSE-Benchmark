import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

def split(
	data: pd.DataFrame,
    training_fraction: float,
    validation_fraction: float,
    testing_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into training, validation, and testing datasets. 
    The training, validation, and testing fractions must sum to 1.0.

    Args:
        data (pd.DataFrame): The dataset to be split.
        training_fraction (float): Fraction of observations in the training dataset.
        validation_fraction (float): Fraction of observations in the validation dataset.
        testing_fraction (float): Fraction of observations in the testing dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the 
			training, validation, and testing DataFrames, respectively.

    Raises:
        ValueError: If the sum of the provided fractions does not equal 1.0.
    """
    # Check split fractions
    total_fraction = training_fraction + validation_fraction + testing_fraction
    if not np.isclose(total_fraction, 1.0, atol=1e-6):
        raise ValueError("Training, validation, and testing fractions must sum to 1.")

    # Determine split end indexes
    total_observations = len(data)
    training_split_end_idx = int(training_fraction * total_observations)
    validation_split_end_idx = training_split_end_idx + int(validation_fraction * total_observations)

    # Split using .iloc for proper DataFrame slicing
    training_split = data.iloc[0:training_split_end_idx]
    validation_split = data.iloc[training_split_end_idx:validation_split_end_idx]
    testing_split = data.iloc[validation_split_end_idx:]

    return training_split, validation_split, testing_split

def window_sampling(
	data: pd.DataFrame,
	input_features: List[str], 
	target_features: List[str], 
	lookback_window: int, 
	horizon_window: int,
	stride: int = 1):

	# Validate window sizes and stride
	if lookback_window <= 0:
		raise ValueError("lookback_window must be a positive integer.")
	if horizon_window <= 0:
		raise ValueError("horizon_window must be a positive integer.")
	if stride <= 0:
		raise ValueError("stride must be a positive integer.")
	
	# Ensure 'datetime' column exists
	if "datetime" not in data.columns:
		raise ValueError("DataFrame must contain a 'datetime' column.")

	# Sort data by datetime to ensure chronological order
	data = data.sort_values('datetime').reset_index(drop = True)

	inputs = data[["datetime"] + input_features]
	targets = data[["datetime"] + target_features]

	# Generate samples
	X, Y = [], []
	dataset = {
		'X': None,
		'Y': None,
		"dates": {
			"first_sample_input_start": None,
			"first_sample_input_end": None,
			"first_sample_target_start": None,
			"first_sample_target_end": None,
			"last_sample_input_start": None,
			"last_sample_input_end": None,
			"last_sample_target_start": None,
			"last_sample_target_end": None
		}
	}

	# Calculate the number of samples
	num_samples = (len(data) - (lookback_window + horizon_window)) // stride + 1
	if num_samples <= 0:
		raise ValueError("Insufficient data to generate even 1 sample with the given lookback and horizon windows.")
	
	# Generate samples
	for sample in range(num_samples):
		t = sample * stride
		input_start = t
		input_end = t + lookback_window
		target_start = input_end
		target_end = target_start + horizon_window

		input = inputs[input_start:input_end]
		target = targets[target_start:target_end]

		if sample == 0: # First sample
			dataset["dates"]["first_input_start_date"] = input["datetime"].min()
			dataset["dates"]["first_input_end_date"] = input["datetime"].max()
			dataset["dates"]["first_target_start_date"] = target["datetime"].min()
			dataset["dates"]["first_target_end_date"] = target["datetime"].max()
		elif sample == num_samples - 1: # Last sample
			dataset["dates"]["last_input_start_date"] = input["datetime"].min()
			dataset["dates"]["last_input_end_date"] = input["datetime"].max()
			dataset["dates"]["last_target_start_date"] = target["datetime"].min()
			dataset["dates"]["last_target_end_date"] = target["datetime"].max()			

		input = input.drop("datetime", axis = 1)
		target = target.drop("datetime", axis = 1)

		# Extract input and target sequences
		X.append(input.values)
		Y.append(target.values)

	# Convert time series samples to NumPy array
	X = np.array(X)
	Y = np.array(Y)

	dataset['X'] = X
	dataset['Y'] = Y

	return dataset

def scale_data(
	data: pd.DataFrame,
	scaler: StandardScaler = None) -> pd.DataFrame:
	
	# Save dates and column names
	columns = data.columns
	dates = data["datetime"]
	dates = dates.reset_index(drop = True)

	# Check if scaler has been provided
	if scaler is not None:
		scaled_data = scaler.transform(data.drop("datetime", axis = 1))
		scaled_data = pd.DataFrame(scaled_data)
	else:
		scaler = StandardScaler()
		scaled_data = scaler.fit_transform(data.drop("datetime", axis = 1))
		scaled_data = pd.DataFrame(scaled_data)

	
	scaled_data = pd.concat([dates, scaled_data], axis = 1)
	scaled_data["datetime"] = scaled_data["datetime"].dt.date
	scaled_data.columns = columns

	return scaled_data, scaler


def prepare_data(
	data: pd.DataFrame,
    training_fraction: float,
    validation_fraction: float,
    testing_fraction: float,
	input_features: List[str], 
	target_features: List[str], 
	lookback_window: int, 
	horizon_window: int,
	stride: int = 1):

	# Split the dataset
	training_data, validation_data, testing_data = split(data, 
													  	 training_fraction, 
														 validation_fraction, 
														 testing_fraction)
	

	# Standardize the data
	training_data, scaler = scale_data(training_data)
	validation_data, _ = scale_data(validation_data, scaler)
	testing_data, _ = scale_data(testing_data, scaler)

	# Perform windowed sampling
	training_data = window_sampling(training_data,
								 	input_features,
									target_features,
									lookback_window,
									horizon_window)
	
	validation_data = window_sampling(validation_data,
									  input_features,
									  target_features,
									  lookback_window,
									  horizon_window)
	
	testing_data = window_sampling(testing_data,
								   input_features,
								   target_features,
								   lookback_window,
								   horizon_window)
	
	return training_data, validation_data, testing_data


	





	






# def sample_and_split(data: pd.DataFrame, 
# 					 input_features: List[str], 
# 					 target_features: List[str], 
# 					 lookback_window: int, 
# 					 horizon_window: int, 
# 					 training_fraction: float, 
# 					 validation_fraction: float, 
# 					 testing_fraction: float, 
# 					 stride: int = 1) -> dict:

# 	# Split data
# 	total_samples = len(X)
# 	train_end = int(total_samples * training_fraction)
# 	val_end = train_end + int(total_samples * validation_fraction)

# 	X_train, Y_train = X[:train_end], Y[:train_end]
# 	X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
# 	X_test, Y_test = X[val_end:], Y[val_end:]

# 	# Function to get the start and end dates for a given split
# 	def get_split_dates(split_start_idx: int, split_end_idx: int) -> Dict[str, pd.Timestamp]:
# 		"""
# 		Retrieves detailed date information for the first and last samples in a given split.

# 		Args:
# 			split_start_idx (int): Starting index of the split.
# 			split_end_idx (int): Ending index of the split.

# 		Returns:
# 			Dict[str, pd.Timestamp]: Contains first and last dates for inputs and targets.
# 		"""
# 		if split_end_idx <= split_start_idx:
# 			# No samples in this split
# 			return {
# 				'first_input_start_date': None,
# 				'first_input_end_date': None,
# 				'first_target_start_date': None,
# 				'first_target_end_date': None,
# 				'last_input_start_date': None,
# 				'last_input_end_date': None,
# 				'last_target_start_date': None,
# 				'last_target_end_date': None
# 			}
		
# 		# First sample in the split
# 		first_sample_idx = split_start_idx
# 		first_input_start = first_sample_idx * stride
# 		first_input_end = first_input_start + lookback_window
# 		first_target_start = first_input_end
# 		first_target_end = first_target_start + horizon_window - 1  # Inclusive
		
# 		# Last sample in the split
# 		last_sample_idx = split_end_idx - 1
# 		last_input_start = last_sample_idx * stride
# 		last_input_end = last_input_start + lookback_window
# 		last_target_start = last_input_end
# 		last_target_end = last_target_start + horizon_window - 1  # Inclusive
		
# 		return {
# 			'first_input_start_date': dates[first_input_start],
# 			'first_input_end_date': dates[first_input_end - 1],
# 			'first_target_start_date': dates[first_target_start],
# 			'first_target_end_date': dates[first_target_end],
# 			'last_input_start_date': dates[last_input_start],
# 			'last_input_end_date': dates[last_input_end - 1],
# 			'last_target_start_date': dates[last_target_start],
# 			'last_target_end_date': dates[last_target_end]
# 		}
	
# 	train = {
# 		'X': X_train,
# 		'Y': Y_train,
# 		"dates": get_split_dates(0, train_end)
# 	}

# 	val = {
# 		'X': X_val,
# 		'Y': Y_val,
# 		"dates": get_split_dates(train_end, val_end)
# 	}

# 	test = {
# 		'X': X_test,
# 		'Y': Y_test,
# 		"dates": get_split_dates(val_end, total_samples)
# 	}
	
# 	return train, val, test









def generate_forecast_matrix(predictions, forecast_horizon) -> List[np.ndarray]:
	"""
	Generates a matrix of forecasts where rows represent days in the forecast horizon,
	and columns represent each datetime in the split.
	
	Args:
		predictions (list): List of numpy arrays of shape (batch_size, forecast_horizon, num_features),
							collected from the last epoch. The predictions should be in the correct
							sequential order (i.e., not shuffled).
		forecast_horizon (int): Number of days ahead the model is predicting (default is 14).
	
	Returns:
		forecast_matrix (numpy.ndarray): Matrix of shape (forecast_horizon, num_date_times)
										  containing the model's forecasts.
	"""

	predictions = np.concatenate(predictions, axis=0)  # Shape: (total_samples, forecast_horizon, num_features)
	num_datetimes = predictions.shape[0]
	num_features = predictions.shape[2]
	
	# Initialize list to hold forecast matrices for each feature
	forecast_matrices = []

	# Loop over each feature to create a forecast matrix for each
	for feature_idx in range(num_features):
		# Initialize an empty forecast matrix for the feature
		forecast_matrix = np.empty((forecast_horizon, num_datetimes))
		forecast_matrix[:] = np.nan  # Fill with NaN initially

		# Populate forecast matrix without averaging overlaps
		for i in range(num_datetimes):
			# Calculate the effective forecast horizon, limiting it if we’re close to the end
			max_horizon = min(forecast_horizon, num_datetimes - i)
			
			# Only fill up to max_horizon, leaving NaNs for rows above
			for day in range(max_horizon):
				forecast_matrix[forecast_horizon - max_horizon + day, i + day] = predictions[i, day, feature_idx]
		
		forecast_matrices.append(forecast_matrix)

	return forecast_matrices
