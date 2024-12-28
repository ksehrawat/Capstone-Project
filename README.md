# Redfin House Price Prediction and Forecasting Project

## Project Overview

The main objective of the project was to explore house prices for the California, New York, and Texas states and analyze market trends for actionable insights. The project also aimed to forecast future house prices using advance machine learning models.

## Files Information

### Data Files

* Initial Dataset: NewRedfinDataSet.csv
* Cleaned Dataset: Redfin_df_cleaned.csv
* Dataset with Prediction (Decision Tree Model): Redfin_df_with_decision_tree_predictions.csv
* Dataset with Prediction (Random Forst Model): Redfin_df_with_predictions_random_forest.csv

### Code File

* Juiter Notebook:

## DataSet

Initial DatSet was downloaded from the Redfin Website (https://www.redfin.com/news/data-center/) and then data was filetered furtner up to focus on the 3 states i.e. California, New York , and Texas. Initial DataSet (NewRedfinDataSet.csv) was stored in the Google Drive Folder so that it can used in Colab for further analysis.


## Data Exploration and Cleaning
### Initial Dataset:

* Rows: 66,577

* Columns: 50

### Key Cleaning Steps:

* Removed missing values and irrelevant columns.

* Handled outliers in median_sale_price using IQR.

* Converted period_begin to datetime format for time-based analysis.

### Final Dataset:

* Rows: 44,723

* Columns: 50

## Modeling Results

### 1. Linear Regression:

* RMSE: $169,626

* R²: 0.725

* Linear model performed well but lacked the ability to capture non-linear relationships.

### 2. Ridge and Lasso Regression:

* Ridge RMSE: $169,637, R²: 0.725

* Lasso RMSE: $169,626, R²: 0.725

* Regularization reduced feature noise but did not improve performance over Random Forest.

### 3. Decision Tree:

* RMSE: $80,617

* R²: 0.938

* Offered interpretable results but showed signs of overfitting.

### 4. Random Forest:

* RMSE: $46,503

* R²: 0.979

* Most accurate model, handling non-linear relationships and outliers effectively.




