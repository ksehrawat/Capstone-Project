# Capstone Project: Redfin House Prediction Project Report

Objective

To build and evaluate a robust model for predicting house prices using historical and feature-rich data from Redfin for the California, Texas, and New York State. The project also involved exploratory analysis, and feature engineering to extract meaningful insights and trends.


**Data Exploration and Cleaning**

Initial Dataset Overview:

Number of rows: ~47,547

Number of columns: 50

Key features: median_sale_price, median_list_price, median_ppsf, homes_sold, new_listings, avg_sale_to_list, etc.

Cleaning Steps:

Handled missing values by dropping columns with >40% missing values and imputing remaining data.

Outliers in median_sale_price were filtered using the IQR method.

Converted period_begin to datetime format for time-based analysis.

Result: A clean dataset ready for modeling and analysis.
