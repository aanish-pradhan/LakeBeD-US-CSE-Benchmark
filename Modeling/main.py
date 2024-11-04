# IMPORT PACKAGES
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from Model.lstm import LSTM
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from Utilities import utilities

# PARSE ARGUMENTS
argparser = argparse.ArgumentParser()

## Data and dataloader arguments
argparser.add_argument("--input_features", type = str, nargs = '+', help = "Input features in the lookback window")
argparser.add_argument("--target_features", type = str, nargs = '+', help = "Target features in the horizon window")
argparser.add_argument("--horizon_window", type = int, default = 14, help = "Days in the future to forecast")
argparser.add_argument("--lookback_window", type = int, default = 21, help = "Days in the past to consider for forecasting")
argparser.add_argument("--batch_size", type = int, default = 32, help = "Number of samples per batch in dataloaders")

## Model architecture arguments
argparser.add_argument("--seed", type = int, default = 42, help = "Seed to use for NumPy and PyTorch")
argparser.add_argument("--encoder_num_recurrent_layers", type = int, default = 1, help = "Number of recurrent layers in the encoder")
argparser.add_argument("--decoder_num_recurrent_layers", type = int, default = 1, help = "Number of recurrent layers in the decoder")
argparser.add_argument("--hidden_state_dimensionality", type = int, default = 8, help = "Dimensionality of the context and hidden state vector")

## Learning hyperparameter arguments
argparser.add_argument("--encoder_dropout_rate", type = float, default = 0.0, help = "Dropout rate of the encoder LSTM-RNN")
argparser.add_argument("--decoder_dropout_rate", type = float, default = 0.0, help = "Dropout rate of the decoder LSTM-RNN")
argparser.add_argument("--num_epochs", type = int, default = 100, help = "Number of epochs in training")
argparser.add_argument("--learning_rate", type = float, default = 1e-3, help = "Initial learning rate")
argparser.add_argument("--lr_scheduler_threshold", type = float, default = 1e-3, help = "Threshold for the learning rate scheduler")
argparser.add_argument("--weight_decay", type = float, default = 0.0, help = "AdaM optimizer weight decay regularization")

args = argparser.parse_args()

# SET SEEDS
seed = args.seed # SEEDS used 42, 40, 64, 51, 50
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# DATA PREPARATION
with open("Training_Data.pickle", "rb") as f:
	train = pickle.load(f)
with open("Validation_Data.pickle", "rb") as f:
	validate = pickle.load(f)
with open("Testing_Data.pickle", "rb") as f:
	test = pickle.load(f)
with open("Standardization_Parameters.pickle", "rb") as f:
	standardization_parameters = pickle.load(f)

train_dataset = TensorDataset(torch.Tensor(train["windowed_data_imputed"]['X']), torch.Tensor(train["windowed_data_imputed"]['Y']))
val_dataset = TensorDataset(torch.Tensor(validate["windowed_data_imputed"]['X']), torch.Tensor(validate["windowed_data_imputed"]['Y']))
test_dataset = TensorDataset(torch.Tensor(test["windowed_data_imputed"]['X']), torch.Tensor(test["windowed_data_imputed"]['Y']))

dataloader_batch_size = args.batch_size
training_loader = DataLoader(train_dataset, batch_size = dataloader_batch_size, shuffle = False)
validation_loader = DataLoader(val_dataset, batch_size = dataloader_batch_size, shuffle = False)
testing_loader = DataLoader(test_dataset, batch_size = dataloader_batch_size, shuffle = False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = LSTM(device = device,
			 hidden_state_dim = args.hidden_state_dimensionality,
			 horizon_window = args.horizon_window,
			 num_input_features = len(args.input_features),
			 num_target_features = len(args.target_features),
			 encoder_num_recurrent_layers = args.encoder_num_recurrent_layers,
			 decoder_num_recurrent_layers = args.decoder_num_recurrent_layers,
			 encoder_dropout_rate = args.encoder_dropout_rate,
			 decoder_dropout_rate = args.decoder_dropout_rate)

# Training parameters
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay

# Train the model
training_predictions, validation_predictions, testing_predictions, training_costs, validation_costs, testing_cost, lr_per_epoch = model.train_model(
	training_loader = training_loader,
	validation_loader = validation_loader,
	test_loader = testing_loader,
	num_epochs = num_epochs,
	learning_rate = learning_rate,
	weight_decay = weight_decay)

# Save results from the model
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(timestamp)

data_summary = {
	"input_features": args.input_features,
	"target_features": args.target_features,

    "train_first_input_start_date": str(train["windowed_data_imputed"]["dates"]["first_input_start_date"]),
    "train_first_input_end_date": str(train["windowed_data_imputed"]["dates"]["first_input_end_date"]),
    "train_first_target_start_date": str(train["windowed_data_imputed"]["dates"]["first_target_start_date"]),
    "train_first_target_end_date": str(train["windowed_data_imputed"]["dates"]["first_target_end_date"]),
    "train_last_input_start_date": str(train["windowed_data_imputed"]["dates"]["last_input_start_date"]),
    "train_last_input_end_date": str(train["windowed_data_imputed"]["dates"]["last_input_end_date"]),
    "train_last_target_start_date": str(train["windowed_data_imputed"]["dates"]["last_target_start_date"]),
    "train_last_target_end_date": str(train["windowed_data_imputed"]["dates"]["last_target_end_date"]),
    
    "validation_first_input_start_date": str(validate["windowed_data_imputed"]["dates"]["first_input_start_date"]),
    "validation_first_input_end_date": str(validate["windowed_data_imputed"]["dates"]["first_input_end_date"]),
    "validation_first_target_start_date": str(validate["windowed_data_imputed"]["dates"]["first_target_start_date"]),
    "validation_first_target_end_date": str(validate["windowed_data_imputed"]["dates"]["first_target_end_date"]),
    "validation_last_input_start_date": str(validate["windowed_data_imputed"]["dates"]["last_input_start_date"]),
    "validation_last_input_end_date": str(validate["windowed_data_imputed"]["dates"]["last_input_end_date"]),
    "validation_last_target_start_date": str(validate["windowed_data_imputed"]["dates"]["last_target_start_date"]),
    "validation_last_target_end_date": str(validate["windowed_data_imputed"]["dates"]["last_target_end_date"]),
    
    "test_first_input_start_date": str(test["windowed_data_imputed"]["dates"]["first_input_start_date"]),
    "test_first_input_end_date": str(test["windowed_data_imputed"]["dates"]["first_input_end_date"]),
    "test_first_target_start_date": str(test["windowed_data_imputed"]["dates"]["first_target_start_date"]),
    "test_first_target_end_date": str(test["windowed_data_imputed"]["dates"]["first_target_end_date"]),
    "test_last_input_start_date": str(test["windowed_data_imputed"]["dates"]["last_input_start_date"]),
    "test_last_input_end_date": str(test["windowed_data_imputed"]["dates"]["last_input_end_date"]),
    "test_last_target_start_date": str(test["windowed_data_imputed"]["dates"]["last_target_start_date"]),
    "test_last_target_end_date": str(test["windowed_data_imputed"]["dates"]["last_target_end_date"])
}

with open(f"{timestamp}/Data_Summary.json", 'w') as f:
	json.dump(data_summary, f, indent = 4)

model_config = {
    "seed": seed,
	"encoder_num_recurrent_layers": args.encoder_num_recurrent_layers,
	"decoder_num_recurrent_layers": args.decoder_num_recurrent_layers,
	"encoder_dropout_rate": args.encoder_dropout_rate,
	"decoder_dropout_rate": args.decoder_dropout_rate,
	"hidden_state_dimensionality": args.hidden_state_dimensionality,
	"num_epochs": num_epochs,
	"learning_rate": args.learning_rate,
}
with open(f"{timestamp}/Model_Config.json", 'w') as f:
	json.dump(model_config, f, indent = 4)


model_results = {
	"final_training_cost": training_costs[-1],
	"final_validation_cost": validation_costs[-1],
	"final_testing_cost": testing_cost
}
with open(f"{timestamp}/Model_Results.json", 'w') as f:
	json.dump(model_results, f, indent = 4)

learning_curve = pd.DataFrame({"epoch": np.arange(1, len(training_costs) + 1),
							   "training_cost": np.array(training_costs),
							   "validation_cost": np.array(validation_costs),
							   "lr_per_epoch": lr_per_epoch})
learning_curve.to_csv(f"{timestamp}/Learning_Curve.csv", index = False)

training_forecast_matrices = utilities.generate_forecast_matrix(training_predictions, args.horizon_window)
validation_forecast_matrices = utilities.generate_forecast_matrix(validation_predictions, args.horizon_window)
testing_forecast_matrices = utilities.generate_forecast_matrix(testing_predictions, args.horizon_window)

# Save training predictions
training_do_predicted = np.nanmedian(training_forecast_matrices[0], axis = 0) * standardization_parameters["do"]["std_dev"] + standardization_parameters["do"]["mean"]
training_do_predicted = np.append(training_do_predicted, [np.nan] * (len(train["original_data"]) - len(training_do_predicted)))
training_temp_predicted = np.nanmedian(training_forecast_matrices[1], axis = 0) * standardization_parameters["temp"]["std_dev"] + standardization_parameters["temp"]["mean"]
temp_predicted = np.append(training_temp_predicted, [np.nan] * (len(train["original_data"]) - len(training_temp_predicted)))

training_forecast = pd.DataFrame({"datetime": train["original_data"]["datetime"],
								  "do_observed": train["original_data"]["do"],
								  "do_predicted": training_do_predicted,
								  "temp_observed": train["original_data"]["temp"],
								  "temp_predicted": temp_predicted})
training_forecast.to_csv(f"{timestamp}/Training_Forecast.csv", index = False)

# Save validation predictions
validation_do_predicted = np.nanmedian(validation_forecast_matrices[0], axis=0) * standardization_parameters["do"]["std_dev"] + standardization_parameters["do"]["mean"]
validation_do_predicted = np.append(validation_do_predicted, [np.nan] * (len(validate["original_data"]) - len(validation_do_predicted)))

validation_temp_predicted = np.nanmedian(validation_forecast_matrices[1], axis=0) * standardization_parameters["temp"]["std_dev"] + standardization_parameters["temp"]["mean"]
temp_predicted = np.append(validation_temp_predicted, [np.nan] * (len(validate["original_data"]) - len(validation_temp_predicted)))

validation_forecast = pd.DataFrame({"datetime": validate["original_data"]["datetime"],
                                    "do_observed": validate["original_data"]["do"],
                                    "do_predicted": validation_do_predicted,
                                    "temp_observed": validate["original_data"]["temp"],
                                    "temp_predicted": temp_predicted})
validation_forecast.to_csv(f"{timestamp}/Validation_Forecast.csv", index=False)

# Save testing preferences

testing_do_predicted = np.nanmean(testing_forecast_matrices[0], axis=0) * standardization_parameters["do"]["std_dev"] + standardization_parameters["do"]["mean"]
testing_do_predicted = np.append(testing_do_predicted, [np.nan] * (len(test["original_data"]) - len(testing_do_predicted)))


testing_temp_predicted = np.nanmean(testing_forecast_matrices[1], axis=0) * standardization_parameters["temp"]["std_dev"] + standardization_parameters["temp"]["mean"]
temp_predicted = np.append(testing_temp_predicted, [np.nan] * (len(test["original_data"]) - len(testing_temp_predicted)))

testing_forecast = pd.DataFrame({"datetime": test["original_data"]["datetime"],
                                 "do_observed": test["original_data"]["do"],
                                 "do_predicted": testing_do_predicted,
                                 "temp_observed": test["original_data"]["temp"],
                                 "temp_predicted": temp_predicted})

testing_forecast.to_csv(f"{timestamp}/Testing_Forecast.csv", index=False)
