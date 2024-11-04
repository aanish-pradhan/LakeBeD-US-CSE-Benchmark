# IMPORT PACKAGES
library(ggplot2)
library(readr)

# DATA INGESTION
trial_1 <- readr::read_csv("Data/Trial-1/Learning_Curve.csv")
trial_2 <- readr::read_csv("Data/Trial-2/Learning_Curve.csv")
trial_3 <- readr::read_csv("Data/Trial-3/Learning_Curve.csv")
trial_4 <- readr::read_csv("Data/Trial-4/Learning_Curve.csv")
trial_5 <- readr::read_csv("Data/Trial-5/Learning_Curve.csv")

# DATA VISUALIZATION

## Training Learning Curve
ggplot() + 
	geom_line(data = trial_1, aes(epoch, training_cost, color = "Trial 1")) + 
	geom_line(data = trial_2, aes(epoch, training_cost, color = "Trial 2")) + 
	geom_line(data = trial_3, aes(epoch, training_cost, color = "Trial 3")) + 
	geom_line(data = trial_4, aes(epoch, training_cost, color = "Trial 4")) + 
	geom_line(data = trial_5, aes(epoch, training_cost, color = "Trial 5")) +
	labs(title = "Learning Curve",
		 subtitle = "Training Cost",
		 x = "Epoch",
		 y = "Cost (RMSE)",
		 color = "Trial") +
	theme_bw()
ggsave("Figures/Training_Learning_Curve.pdf", height = 6, width = 15, 
	   units = "in", scale = 0.8)

## Validation Learning Curve
ggplot() + 
	geom_line(data = trial_1, aes(epoch, validation_cost, color = "Trial 1")) + 
	geom_line(data = trial_2, aes(epoch, validation_cost, color = "Trial 2")) + 
	geom_line(data = trial_3, aes(epoch, validation_cost, color = "Trial 3")) + 
	geom_line(data = trial_4, aes(epoch, validation_cost, color = "Trial 4")) + 
	geom_line(data = trial_5, aes(epoch, validation_cost, color = "Trial 5")) +
	labs(title = "Learning Curve",
		 subtitle = "Validation Cost",
		 x = "Epoch",
		 y = "Cost (RMSE)",
		 color = "Trial") +
	theme_bw()
ggsave("Figures/Validation_Learning_Curve.pdf", height = 6, width = 15, 
	   units = "in", scale = 0.8)
