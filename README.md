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
     Column                          Non-Null Count   Dtype  
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
``` python
# Print descriptive statistics for all numeric columns

print("\nDescriptive Statistics for Numeric Columns:\n")
print(Redfin_df.describe().to_markdown(numalign="left", stralign="left"))

# For all object type columns, print the number of distinct values and the most frequent value

print("\nObject Column Summaries:\n")
for col in Redfin_df.select_dtypes(include='object'):
    print(f"Column: {col}")
    print(f"  Number of distinct values: {Redfin_df[col].nunique()}")
    print(f"  Most frequent value: {Redfin_df[col].mode()[0]}\n")
```
Descriptive Statistics for Numeric Columns:

|       | period_duration   | region_type_id   | table_id   | ZipCode   | property_type_id   | median_sale_price   | median_sale_price_mom   | median_sale_price_yoy   | median_list_price   | median_list_price_mom   | median_list_price_yoy   | median_ppsf   | median_ppsf_mom   | median_ppsf_yoy   | median_list_ppsf   | median_list_ppsf_mom   | median_list_ppsf_yoy   | homes_sold   | homes_sold_mom   | homes_sold_yoy   | pending_sales   | pending_sales_mom   | pending_sales_yoy   | new_listings   | new_listings_mom   | new_listings_yoy   | inventory   | inventory_mom   | inventory_yoy   | median_dom   | median_dom_mom   | median_dom_yoy   | avg_sale_to_list   | avg_sale_to_list_mom   | avg_sale_to_list_yoy   | sold_above_list   | sold_above_list_mom   | sold_above_list_yoy   | off_market_in_two_weeks   | off_market_in_two_weeks_mom   | off_market_in_two_weeks_yoy   | parent_metro_region_metro_code   |
|:------|:------------------|:-----------------|:-----------|:----------|:-------------------|:--------------------|:------------------------|:------------------------|:--------------------|:------------------------|:------------------------|:--------------|:------------------|:------------------|:-------------------|:-----------------------|:-----------------------|:-------------|:-----------------|:-----------------|:----------------|:--------------------|:--------------------|:---------------|:-------------------|:-------------------|:------------|:----------------|:----------------|:-------------|:-----------------|:-----------------|:-------------------|:-----------------------|:-----------------------|:------------------|:----------------------|:----------------------|:--------------------------|:------------------------------|:------------------------------|:---------------------------------|
| count | 66577             | 66577            | 66577      | 66577     | 66577              | 66577               | 63709                   | 59774                   | 62181               | 59190                   | 56409                   | 66135         | 63243             | 59251             | 61927              | 58917                  | 56110                  | 66577        | 63709            | 59774            | 63659           | 60670               | 57412               | 62192          | 59201              | 56416              | 59124       | 55935           | 53683           | 66194        | 63306            | 59259            | 64080              | 61227                  | 57191                  | 66577             | 63709                 | 59774                 | 63659                     | 60670                         | 57412                         | 66577                            |
| mean  | 90                | 2                | 26277      | 63015     | 3.76732            | 609558              | 0.0245059               | 0.16535                 | 634568              | 0.0368036               | 0.205783                | 603.933       | 0.104704          | 0.306517          | 574.005            | 0.0685711              | 0.410416               | 31.7333      | 0.0545225        | 0.148366         | 35.6847         | 0.0409025           | 0.117801            | 40.6555        | 0.0576317          | 0.161157           | 28.6715     | 0.0766926       | 0.213494        | 55.6175      | 0.146084         | -4.91824         | 0.996805           | -4.70314e-05           | 0.00238509             | 0.356858          | -0.000323411          | 0.0118854             | 0.362377                  | -0.00205065                   | 0.00567192                    | 31650.6                          |
| std   | 0                 | 0                | 15101.8    | 35023.7   | 4.2088             | 741121              | 0.287054                | 2.88463                 | 690112              | 1.46421                 | 8.55913                 | 47153.2       | 12.2758           | 15.0063           | 32215.2            | 6.1679                 | 27.403                 | 52.0972      | 0.412052         | 1.21334          | 56.6844         | 0.37946             | 1.10001             | 63.788         | 0.422659           | 1.22871            | 47.7315     | 0.464358        | 1.17789         | 110.484      | 71.337           | 111.306          | 0.0572743          | 0.028953               | 0.0565173              | 0.307632          | 0.15478               | 0.310974              | 0.300212                  | 0.154288                      | 0.275517                      | 10515.4                          |
| min   | 90                | 2                | 2277       | 6390      | -1                 | 750                 | -0.991111               | -0.994444               | 700                 | -0.995472               | -0.997159               | 0.0455761     | -0.999623         | -0.999643         | 0.550314           | -0.999233              | -0.99925               | 1            | -0.857143        | -0.958333        | 1               | -0.875              | -0.972973           | 1              | -0.923077          | -0.977273          | 1           | -0.933333       | -0.961538       | 1            | -3014            | -6624            | 0.5                | -0.795299              | -0.989951              | 0                 | -1                    | -1                    | 0                         | -1                            | -1                            | 10180                            |
| 25%   | 90                | 2                | 4938       | 13820     | -1                 | 249950              | -0.0272667              | -0.0402095              | 262250              | -0.0304331              | -0.029304               | 153.374       | -0.0197275        | -0.0200437        | 160.419            | -0.0207407             | -0.00934908            | 3            | -0.125           | -0.322581        | 4               | -0.129032           | -0.306818           | 4              | -0.142857          | -0.271042          | 4           | -0.139388       | -0.333333       | 20           | -4               | -19              | 0.972973           | -0.00742196            | -0.0224034             | 0.0151515         | -0.0361842            | -0.125                | 0.0625                    | -0.0429009                    | -0.115385                     | 23620                            |
| 50%   | 90                | 2                | 33525      | 77571     | 4                  | 435000              | 0                       | 0.0820896               | 454450              | 0                       | 0.0733029               | 258.865       | 0                 | 0.0845109         | 272.906            | 0                      | 0.075194               | 10           | 0                | -0.0333333       | 12              | 0                   | -0.0588235          | 15             | 0                  | 0                  | 12          | 0               | 0               | 35.5         | 0                | -1.5             | 0.996033           | 0                      | 0.0030105              | 0.333333          | 0                     | 0                     | 0.333333                  | 0                             | 0                             | 35004                            |
| 75%   | 90                | 2                | 38239      | 92262     | 6                  | 749000              | 0.0411311               | 0.219439                | 769500              | 0.0422489               | 0.194223                | 471.287       | 0.0327817         | 0.205783          | 484.261            | 0.0319239              | 0.177115               | 40           | 0.135135         | 0.259259         | 46              | 0.129575            | 0.222222            | 52             | 0.166667           | 0.277778           | 34          | 0.172414        | 0.4             | 62.5         | 4.5              | 13               | 1.01887            | 0.00714243             | 0.0268262              | 0.571429          | 0.035102              | 0.166667              | 0.571429                  | 0.0367893                     | 0.127714                      | 40380                            |
| max   | 90                | 2                | 42009      | 96162     | 13                 | 6.59e+07            | 18.076                  | 393.737                 | 3.65e+07            | 249.292                 | 1853.27                 | 1.2033e+07    | 2528.75           | 1946              | 7.75005e+06        | 1045.88                | 5123.97                | 934          | 8                | 113              | 857             | 8                   | 81                  | 995            | 19                 | 78                 | 767         | 15              | 69              | 6715         | 6678             | 3705.5           | 1.97995            | 0.495689               | 0.99709                | 1                 | 1                     | 1                     | 1                         | 1                             | 1                             | 49820                            |

Object Column Summaries:

Column: period_begin
Number of distinct values: 42
Most frequent value: 7/1/21

Column: period_end
Number of distinct values: 42
Most frequent value: 9/30/21

Column: region_type
Number of distinct values: 1
Most frequent value: zip code

Column: is_seasonally_adjusted
Number of distinct values: 1
Most frequent value: f

Column: state
Number of distinct values: 3
Most frequent value: California

Column: state_code
Number of distinct values: 3
Most frequent value: CA

Column: property_type
Number of distinct values: 5
Most frequent value: All Residential

Column: parent_metro_region
Number of distinct values: 136
Most frequent value: Los Angeles, CA

```python
# Print the count and percentage of missing values for each column
missing_values = Redfin_df.isnull().sum()
missing_percent = (missing_values / len(Redfin_df)) * 100
print("Missing Values:\n")
print(pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percentage']).sort_values(by='Count', ascending=False).to_markdown(numalign="left", stralign="left")
```
Missing Values:

|                                | Count   | Percentage   |
|:-------------------------------|:--------|:-------------|
| inventory_yoy                  | 12894   | 19.367       |
| inventory_mom                  | 10642   | 15.9845      |
| median_list_ppsf_yoy           | 10467   | 15.7216      |
| median_list_price_yoy          | 10168   | 15.2725      |
| new_listings_yoy               | 10161   | 15.262       |
| avg_sale_to_list_yoy           | 9386    | 14.098       |
| off_market_in_two_weeks_yoy    | 9165    | 13.766       |
| pending_sales_yoy              | 9165    | 13.766       |
| median_list_ppsf_mom           | 7660    | 11.5055      |
| inventory                      | 7453    | 11.1946      |
| median_list_price_mom          | 7387    | 11.0954      |
| new_listings_mom               | 7376    | 11.0789      |
| median_ppsf_yoy                | 7326    | 11.0038      |
| median_dom_yoy                 | 7318    | 10.9918      |
| median_sale_price_yoy          | 6803    | 10.2182      |
| sold_above_list_yoy            | 6803    | 10.2182      |
| homes_sold_yoy                 | 6803    | 10.2182      |
| pending_sales_mom              | 5907    | 8.87243      |
| off_market_in_two_weeks_mom    | 5907    | 8.87243      |
| avg_sale_to_list_mom           | 5350    | 8.03581      |
| median_list_ppsf               | 4650    | 6.98439      |
| median_list_price              | 4396    | 6.60288      |
| new_listings                   | 4385    | 6.58636      |
| median_ppsf_mom                | 3334    | 5.00774      |
| median_dom_mom                 | 3271    | 4.91311      |
| off_market_in_two_weeks        | 2918    | 4.38289      |
| pending_sales                  | 2918    | 4.38289      |
| median_sale_price_mom          | 2868    | 4.30779      |
| sold_above_list_mom            | 2868    | 4.30779      |
| homes_sold_mom                 | 2868    | 4.30779      |
| avg_sale_to_list               | 2497    | 3.75054      |
| median_ppsf                    | 442     | 0.663893     |
| median_dom                     | 383     | 0.575274     |
| period_begin                   | 0       | 0            |
| property_type_id               | 0       | 0            |
| property_type                  | 0       | 0            |
| median_sale_price              | 0       | 0            |
| table_id                       | 0       | 0            |
| is_seasonally_adjusted         | 0       | 0            |
| ZipCode                        | 0       | 0            |
| state                          | 0       | 0            |
| state_code                     | 0       | 0            |
| region_type                    | 0       | 0            |
| region_type_id                 | 0       | 0            |
| period_duration                | 0       | 0            |
| period_end                     | 0       | 0            |
| homes_sold                     | 0       | 0            |
| sold_above_list                | 0       | 0            |
| parent_metro_region            | 0       | 0            |
| parent_metro_region_metro_code | 0       | 0            |

```python
# Drop all rows with missing values

Redfin_df_cleaned = Redfin_df.dropna()

# Display updated dataset info and first few rows
Redfin_df_dropped_info = Redfin_df_cleaned.info()
Redfin_df_dropped_preview = print(Redfin_df_cleaned.head().to_markdown(index=False, numalign="left", stralign="left"))

Redfin_df_dropped_info, Redfin_df_dropped_preview
```
Data columns (total 50 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   period_begin                    47547 non-null  object 
 1   period_end                      47547 non-null  object 
 2   period_duration                 47547 non-null  int64  
 3   region_type                     47547 non-null  object 
 4   region_type_id                  47547 non-null  int64  
 5   table_id                        47547 non-null  int64  
 6   is_seasonally_adjusted          47547 non-null  object 
 7   ZipCode                         47547 non-null  int64  
 8   state                           47547 non-null  object 
 9   state_code                      47547 non-null  object 
 10  property_type                   47547 non-null  object 
 11  property_type_id                47547 non-null  int64  
 12  median_sale_price               47547 non-null  float64
 13  median_sale_price_mom           47547 non-null  float64
 14  median_sale_price_yoy           47547 non-null  float64
 15  median_list_price               47547 non-null  float64
 16  median_list_price_mom           47547 non-null  float64
 17  median_list_price_yoy           47547 non-null  float64
 18  median_ppsf                     47547 non-null  float64
 19  median_ppsf_mom                 47547 non-null  float64
 20  median_ppsf_yoy                 47547 non-null  float64
 21  median_list_ppsf                47547 non-null  float64
 22  median_list_ppsf_mom            47547 non-null  float64
 23  median_list_ppsf_yoy            47547 non-null  float64
 24  homes_sold                      47547 non-null  int64  
 25  homes_sold_mom                  47547 non-null  float64
 26  homes_sold_yoy                  47547 non-null  float64
 27  pending_sales                   47547 non-null  float64
 28  pending_sales_mom               47547 non-null  float64
 29  pending_sales_yoy               47547 non-null  float64
 30  new_listings                    47547 non-null  float64
 31  new_listings_mom                47547 non-null  float64
 32  new_listings_yoy                47547 non-null  float64
 33  inventory                       47547 non-null  float64
 34  inventory_mom                   47547 non-null  float64
 35  inventory_yoy                   47547 non-null  float64
 36  median_dom                      47547 non-null  float64
 37  median_dom_mom                  47547 non-null  float64
 38  median_dom_yoy                  47547 non-null  float64
 39  avg_sale_to_list                47547 non-null  float64
 40  avg_sale_to_list_mom            47547 non-null  float64
 41  avg_sale_to_list_yoy            47547 non-null  float64
 42  sold_above_list                 47547 non-null  float64
 43  sold_above_list_mom             47547 non-null  float64
 44  sold_above_list_yoy             47547 non-null  float64
 45  off_market_in_two_weeks         47547 non-null  float64
 46  off_market_in_two_weeks_mom     47547 non-null  float64
 47  off_market_in_two_weeks_yoy     47547 non-null  float64
 48  parent_metro_region             47547 non-null  object 
 49  parent_metro_region_metro_code  47547 non-null  int64  
dtypes: float64(35), int64(7), object(8)
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






