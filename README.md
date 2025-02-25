# Redfin House Price Prediction and Forecasting Project

## Project Overview

The real estate market is dynamic and influenced by various factors such as location, property size, neighborhood quality, and proximity to amenities. Accurately predicting house prices is crucial for buyers, sellers, investors, and financial institutions to make informed decisions. The challenge lies in identifying key drivers of house prices and building a robust predictive model that can generalize well across different markets.

## Research Question

How can we leverage historical housing data and machine learning models to accurately predict house prices, identify key influencing factors, and provide actionable insights for market participants?

## Files Information

### Data Files

* Initial Dataset: NewRedfinDataSet.csv
* Cleaned Dataset: Redfin_df_cleaned.csv

### Python Code File

* Juiter Notebook: Capstone.ipynb

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

## Data Visualization

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




