import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np
import random
import os
import pickle
import json
from Model.lstm import LSTM

def objective(trial):
    # Hyperparameters to optimize
    hidden_state_dim = trial.suggest_int("hidden_state_dim", 4, 64)
    num_recurrent_layers = trial.suggest_int("num_recurrent_layers", 1, 3)  # Use the same number for both encoder and decoder
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Adjust dropout rate if num_layers is 1 (avoid warnings)
    encoder_dropout_rate = dropout_rate if num_recurrent_layers > 1 else 0.0
    decoder_dropout_rate = dropout_rate if num_recurrent_layers > 1 else 0.0

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Load data (adjust paths as necessary)
    with open("Training_Data.pickle", "rb") as f:
        train = pickle.load(f)
    with open("Validation_Data.pickle", "rb") as f:
        validate = pickle.load(f)
    with open("Testing_Data.pickle", "rb") as f:
        test = pickle.load(f)

    train_dataset = TensorDataset(torch.Tensor(train["windowed_data_imputed"]['X']), torch.Tensor(train["windowed_data_imputed"]['Y']))
    val_dataset = TensorDataset(torch.Tensor(validate["windowed_data_imputed"]['X']), torch.Tensor(validate["windowed_data_imputed"]['Y']))
    testing_dataset = TensorDataset(torch.Tensor(test["windowed_data_imputed"]['X']), torch.Tensor(test["windowed_data_imputed"]['Y']))
    dataloader_batch_size = 32
    training_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=False)
    validation_loader = DataLoader(val_dataset, batch_size=dataloader_batch_size, shuffle=False)
    testing_loader = DataLoader(testing_dataset, batch_size = dataloader_batch_size, shuffle = False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model with trial hyperparameters
    model = LSTM(
        device=device,
        hidden_state_dim=hidden_state_dim,
        horizon_window=14,  # fixed horizon window
        num_input_features=len(train["windowed_data_imputed"]['X'][0][0]),  # infer input features from data
        num_target_features=len(train["windowed_data_imputed"]['Y'][0][0]),  # infer target features from data
        encoder_dropout_rate=encoder_dropout_rate,
        decoder_dropout_rate=decoder_dropout_rate,
        encoder_num_recurrent_layers=num_recurrent_layers,
        decoder_num_recurrent_layers=num_recurrent_layers
    )

    # Define training parameters
    num_epochs = 20  # keep it low for faster trial runs
    early_stopping_patience = 3

    # Train the model and capture validation performance
    _, validation_predictions, _, _, average_validation_costs, _, _ = model.train_model(
        training_loader=training_loader,
        validation_loader=validation_loader,
        test_loader = testing_loader,  # Optuna doesn’t evaluate test cost, only validation
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience
    )

    # Return the best validation loss for Optuna to minimize
    best_validation_cost = min(average_validation_costs)
    return best_validation_cost

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # Adjust number of trials as needed

# Print best trial
print("Best trial:")
trial = study.best_trial

print(f"  Value (Best Validation Cost): {trial.value}")
print("  Params (Best Hyperparameters):")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save the best model hyperparameters
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(timestamp)
with open(f"{timestamp}/Best_Hyperparameters.json", "w") as f:
    json.dump(trial.params, f, indent=4)
