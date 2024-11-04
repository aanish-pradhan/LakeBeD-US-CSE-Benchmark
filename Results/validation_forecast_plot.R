# IMPORT PACKAGES
library(ggplot2)
library(latex2exp)
library(readr)

# DATA INGESTION
trial_1 <- readr::read_csv("Data/Trial-1/Validation_Forecast.csv")
trial_2 <- readr::read_csv("Data/Trial-2/Validation_Forecast.csv")
trial_3 <- readr::read_csv("Data/Trial-3/Validation_Forecast.csv")
trial_4 <- readr::read_csv("Data/Trial-4/Validation_Forecast.csv")
trial_5 <- readr::read_csv("Data/Trial-5/Validation_Forecast.csv")

# DATA WRANGLING

do_forecasts <- data.frame(
	trial_1 = trial_1$do_predicted,
	trial_2 = trial_2$do_predicted,
	trial_3 = trial_3$do_predicted,
	trial_4 = trial_4$do_predicted,
	trial_5 = trial_5$do_predicted)
do_forecasts$do_min <- as.numeric(apply(do_forecasts, 1, min))
do_forecasts$do_median <- as.numeric(apply(do_forecasts, 1, median))
do_forecasts$do_max <- as.numeric(apply(do_forecasts, 1, max))
do_forecasts$datetime <- trial_1$datetime

temp_forecasts <- data.frame(
	trial_1 = trial_1$temp_predicted,
	trial_2 = trial_2$temp_predicted,
	trial_3 = trial_3$temp_predicted,
	trial_4 = trial_4$temp_predicted,
	trial_5 = trial_5$temp_predicted)
temp_forecasts$temp_min <- as.numeric(apply(temp_forecasts, 1, min))
temp_forecasts$temp_median <- as.numeric(apply(temp_forecasts, 1, median))
temp_forecasts$temp_max <- as.numeric(apply(temp_forecasts, 1, max))
temp_forecasts$datetime <- trial_1$datetime

# DATA VISUALIZATION

## Dissolved Oxygen Forecast
ggplot() + 
	geom_line(data = trial_1, aes(datetime, do_observed, 
		color = "Observed")) + 
	geom_line(data = do_forecasts, aes(datetime, do_median, 
		color = "Predicted")) + 
	geom_ribbon(data = do_forecasts, aes(datetime, ymin = do_min, 
		ymax = do_max, fill = "Prediction Confidence Interval"), alpha = 0.5) + 
	labs(title = "Dissolved Oxygen Concentration Forecast",
		 subtitle = "Validation Split: May 2020 - February 2022",
		 x = "Date",
		 y = latex2exp::TeX("Concentration $\\left(\\frac{mg}{L} \\right)$")) + 
	theme_bw()
ggsave("Figures/Validation_Forecast_Dissolved_Oxygen_Concentration.pdf", 
	height = 6, width = 15, scale = 0.8)

## Water Temperature Forecast
ggplot() + 
	geom_line(data = trial_1, aes(datetime, temp_observed, 
		color = "Observed")) + 
	geom_line(data = temp_forecasts, aes(datetime, temp_median, 
		color = "Predicted")) + 
	geom_ribbon(data = temp_forecasts, aes(datetime, ymin = temp_min, 
		ymax = temp_max, fill = "Prediction Confidence Interval"), 
		alpha = 0.5) + 
	labs(title = "Water Temperature Forecast",
		 subtitle = "Validation Split: May 2020 - February 2022",
		 x = "Date",
		 y = expression("Temperature ("*degree*"C)")) + 
	theme_bw()
ggsave("Figures/Validation_Forecast_Water_Temperature.pdf", height = 6, 
	   width = 15, scale = 0.8)
