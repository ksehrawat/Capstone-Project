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

## Create DataFrame in Python for the Data Set
```python

file_path = '/content/drive/MyDrive/AI Capstone Project/NewRedfinDataSet.csv'
Redfin_df = pd.read_csv(file_path)

print(Redfin_df.head().to_markdown(index=False, numalign="left", stralign="left"))

Redfin_df.info()
```
Data columns (total 50 columns):
 #   Column                          Non-Null Count   Dtype  
---  ------                          --------------   -----  
 0   period_begin                    66577 non-null   object 
 1   period_end                      66577 non-null   object 
 2   period_duration                 66577 non-null   int64  
 3   region_type                     66577 non-null   object 
 4   region_type_id                  66577 non-null   int64  
 5   table_id                        66577 non-null   int64  
 6   is_seasonally_adjusted          66577 non-null   object 
 7   ZipCode                         66577 non-null   int64  
 8   state                           66577 non-null   object 
 9   state_code                      66577 non-null   object 
 10  property_type                   66577 non-null   object 
 11  property_type_id                66577 non-null   int64  
 12  median_sale_price               66577 non-null   float64
 13  median_sale_price_mom           63709 non-null   float64
 14  median_sale_price_yoy           59774 non-null   float64
 15  median_list_price               62181 non-null   float64
 16  median_list_price_mom           59190 non-null   float64
 17  median_list_price_yoy           56409 non-null   float64
 18  median_ppsf                     66135 non-null   float64
 19  median_ppsf_mom                 63243 non-null   float64
 20  median_ppsf_yoy                 59251 non-null   float64
 21  median_list_ppsf                61927 non-null   float64
 22  median_list_ppsf_mom            58917 non-null   float64
 23  median_list_ppsf_yoy            56110 non-null   float64
 24  homes_sold                      66577 non-null   int64  
 25  homes_sold_mom                  63709 non-null   float64
 26  homes_sold_yoy                  59774 non-null   float64
 27  pending_sales                   63659 non-null   float64
 28  pending_sales_mom               60670 non-null   float64
 29  pending_sales_yoy               57412 non-null   float64
 30  new_listings                    62192 non-null   float64
 31  new_listings_mom                59201 non-null   float64
 32  new_listings_yoy                56416 non-null   float64
 33  inventory                       59124 non-null   float64
 34  inventory_mom                   55935 non-null   float64
 35  inventory_yoy                   53683 non-null   float64
 36  median_dom                      66194 non-null   float64
 37  median_dom_mom                  63306 non-null   float64
 38  median_dom_yoy                  59259 non-null   float64
 39  avg_sale_to_list                64080 non-null   float64
 40  avg_sale_to_list_mom            61227 non-null   float64
 41  avg_sale_to_list_yoy            57191 non-null   float64
 42  sold_above_list                 66577 non-null   float64
 43  sold_above_list_mom             63709 non-null   float64
 44  sold_above_list_yoy             59774 non-null   float64
 45  off_market_in_two_weeks         63659 non-null   float64
 46  off_market_in_two_weeks_mom     60670 non-null   float64
 47  off_market_in_two_weeks_yoy     57412 non-null   float64
 48  parent_metro_region             66577 non-null   object 
 49  parent_metro_region_metro_code  66577 non-null   int64  
dtypes: float64(35), int64(7), object(8)

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






