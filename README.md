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

### Model Training and Evaluation
Several machine learning models were trained and evaluated based on Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

Model Performance Comparison

<img width="602" alt="Screenshot 2025-03-16 at 2 09 31 PM" src="https://github.com/user-attachments/assets/afbd006f-ef0a-4528-b7aa-ccd1c5e9bbc8" />

Key Insights:
* Random Forest performed best with the lowest RMSE (82,251) and highest R² (0.935).
* Gradient Boosting was a close second (RMSE: 83,438, R²: 0.933).
* Linear models (Ridge, Lasso, and Standard Linear Regression) performed poorly due to the complex relationships in housing data.

## DataSet

Initial DatSet was downloaded from the Redfin Website (https://www.redfin.com/news/data-center/) and then data was filetered furtner up to focus on the 3 states i.e. California, New York , and Texas. Initial DataSet (NewRedfinDataSet.csv) was stored in the Google Drive Folder so that it can used in Colab for further analysis.

## Create DataFrame in Python for the Data Set
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
sns.set(style="whitegrid")

file_path = '/content/drive/MyDrive/AI Capstone Project/NewRedfinDataSet.csv'
Redfin_df = pd.read_csv(file_path)

print(Redfin_df.head().to_markdown(index=False, numalign="left", stralign="left"))

Redfin_df.info()
```
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

### Object Column Summaries:

#### Column: period_begin
* Number of distinct values: 42
* Most frequent value: 7/1/21

#### Column: period_end
* Number of distinct values: 42
* Most frequent value: 9/30/21

#### Column: region_type
* Number of distinct values: 1
* Most frequent value: zip code

#### Column: is_seasonally_adjusted
* Number of distinct values: 1
* Most frequent value: f

#### Column: state
* Number of distinct values: 3
* Most frequent value: California

#### Column: state_code
* Number of distinct values: 3
* Most frequent value: CA

#### Column: property_type
* Number of distinct values: 5
* Most frequent value: All Residential

#### Column: parent_metro_region
* Number of distinct values: 136
* Most frequent value: Los Angeles, CA

```python
# Print the count and percentage of missing values for each column
missing_values = Redfin_df.isnull().sum()
missing_percent = (missing_values / len(Redfin_df)) * 100
print("Missing Values:\n")
print(pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percentage']).sort_values(by='Count', ascending=False).to_markdown(numalign="left", stralign="left")
```
Missing Values:

| Column Name                    | Count   | Percentage   |
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
```python
# Print the count and percentage of missing values for each column
missing_values = Redfin_df_cleaned.isnull().sum()
missing_percent = (missing_values / len(Redfin_df_cleaned)) * 100
print("Missing Values:\n")
print(pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percentage']).sort_values(by='Count', ascending=False).to_markdown(numalign="left", stralign="left"))
```
Missing Values:

|                                | Count   | Percentage   |
|:-------------------------------|:--------|:-------------|
| period_begin                   | 0       | 0            |
| median_dom_mom                 | 0       | 0            |
| pending_sales                  | 0       | 0            |
| pending_sales_mom              | 0       | 0            |
| pending_sales_yoy              | 0       | 0            |
| new_listings                   | 0       | 0            |
| new_listings_mom               | 0       | 0            |
| new_listings_yoy               | 0       | 0            |
| inventory                      | 0       | 0            |
| inventory_mom                  | 0       | 0            |
| inventory_yoy                  | 0       | 0            |
| median_dom                     | 0       | 0            |
| median_dom_yoy                 | 0       | 0            |
| period_end                     | 0       | 0            |
| avg_sale_to_list               | 0       | 0            |
| avg_sale_to_list_mom           | 0       | 0            |
| avg_sale_to_list_yoy           | 0       | 0            |
| sold_above_list                | 0       | 0            |
| sold_above_list_mom            | 0       | 0            |
| sold_above_list_yoy            | 0       | 0            |
| off_market_in_two_weeks        | 0       | 0            |
| off_market_in_two_weeks_mom    | 0       | 0            |
| off_market_in_two_weeks_yoy    | 0       | 0            |
| parent_metro_region            | 0       | 0            |
| homes_sold_yoy                 | 0       | 0            |
| homes_sold_mom                 | 0       | 0            |
| homes_sold                     | 0       | 0            |
| median_list_ppsf_yoy           | 0       | 0            |
| period_duration                | 0       | 0            |
| region_type                    | 0       | 0            |
| region_type_id                 | 0       | 0            |
| table_id                       | 0       | 0            |
| is_seasonally_adjusted         | 0       | 0            |
| ZipCode                        | 0       | 0            |
| state                          | 0       | 0            |
| state_code                     | 0       | 0            |
| property_type                  | 0       | 0            |
| property_type_id               | 0       | 0            |
| median_sale_price              | 0       | 0            |
| median_sale_price_mom          | 0       | 0            |
| median_sale_price_yoy          | 0       | 0            |
| median_list_price              | 0       | 0            |
| median_list_price_mom          | 0       | 0            |
| median_list_price_yoy          | 0       | 0            |
| median_ppsf                    | 0       | 0            |
| median_ppsf_mom                | 0       | 0            |
| median_ppsf_yoy                | 0       | 0            |
| median_list_ppsf               | 0       | 0            |
| median_list_ppsf_mom           | 0       | 0            |
| parent_metro_region_metro_code | 0       | 0            |

#### The dataset now contains only rows without missing values:
* Total Rows: 47,547
* Columns: 50


```python
# Remove Outliers from the Dataset
Q1 = Redfin_df_cleaned['median_sale_price'].quantile(0.25)
Q3 = Redfin_df_cleaned['median_sale_price'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
Redfin_df_cleaned = Redfin_df_cleaned[(Redfin_df_cleaned['median_sale_price'] >= lower_bound) &
                        (Redfin_df_cleaned['median_sale_price'] <= upper_bound)]
```

#### Outliers in median_sale_price were removed based on the Interquartile Range (IQR) method.
* Total Rows: 44723
* Median_sale_price Statistics:
* Minimum: $1,800
* Maximum: $1,521,000
* Mean: $522,755
* Median: $442,000

```python
# Exporting and Saving Clean Dataset to Google Share Drive
export_path = '/content/drive/MyDrive/AI Capstone Project/Redfin_df_cleaned.csv'
Redfin_df_cleaned.to_csv(export_path, index=False)
```

#### Data Cleaning Summary:

#### Initial Dataset:
* Rows: 66,577
* Columns: 50

#### Key Cleaning Steps:
* Removed missing values and irrelevant columns.
* Handled outliers in median_sale_price using IQR.
* Converted period_begin to datetime format for time-based analysis.

#### Final Dataset:
* Rows: 44,723
* Columns: 50

## Data Visualization
#### Visualization 1: Distribution of Median Sale Price
```python
# Visualization 1: Distribution of Median Sale Price
plt.figure(figsize=(10, 6))
sns.histplot(Redfin_df_cleaned['median_sale_price'], kde=True, bins=30, color='blue')
plt.title("Distribution of Median Sale Price")
plt.xlabel("Median Sale Price")
plt.ylabel("Frequency")
plt.show()
```
<img width="879" alt="Screenshot 2025-02-24 at 8 55 59 PM" src="https://github.com/user-attachments/assets/20e0be76-2759-4592-99f3-0f03a532bc4a" />

#### Visualization 2: Median Sale Price by State
```python
# Visualization 2: Median Sale Price by State
plt.figure(figsize=(12, 8))
state_price = Redfin_df_cleaned.groupby('state')['median_sale_price'].median().sort_values()
sns.barplot(y=state_price.index, x=state_price.values, palette="viridis", legend=False, hue = state_price.index)
plt.title("Median Sale Price by State")
plt.xlabel("Median Sale Price")
plt.ylabel("State")
plt.show()
```
<img width="1076" alt="Screenshot 2025-02-24 at 8 58 08 PM" src="https://github.com/user-attachments/assets/a7908efc-de08-46ee-ab7a-30a66d6e7f98" />

#### Visualization 3: Median Sale Price vs Homes Sold
```python
# Visualization 3: Median Sale Price vs Homes Sold
plt.figure(figsize=(10, 6))
sns.scatterplot(data=Redfin_df_cleaned, x='homes_sold', y='median_sale_price', hue='state', alpha=0.6)
plt.title("Median Sale Price vs Homes Sold")
plt.xlabel("Homes Sold")
plt.ylabel("Median Sale Price")
plt.legend([], [], frameon=False)  # Hide legend for clarity
plt.show()
```
<img width="859" alt="Screenshot 2025-02-24 at 9 00 55 PM" src="https://github.com/user-attachments/assets/26d2cdaf-fbc6-4daa-b739-12969c4f2f85" />

#### Visualization 4: Trend of Median Sale Price Over Time
```python
# Visualization 4: Trend of Median Sale Price Over Time

# Convert `period_begin` to datetime with specified format
Redfin_df_cleaned['period_begin'] = pd.to_datetime(Redfin_df_cleaned['period_begin'], format='%m/%d/%y', errors='coerce')

# Trend of Median Sale Price Over Time
time_trend_updated = Redfin_df_cleaned.groupby('period_begin')['median_sale_price'].median()
plt.figure(figsize=(12, 6))
time_trend_updated.plot(marker='o', color='green')
plt.title("Trend of Median Sale Price Over Time")
plt.xlabel("Time")
plt.ylabel("Median Sale Price")
plt.grid(True)
plt.show()
```
<img width="1075" alt="Screenshot 2025-02-24 at 9 05 51 PM" src="https://github.com/user-attachments/assets/9613471b-e03c-4677-a738-4fd57d507b1f" />

#### Visualization 5: Boxplot of Median Sale Price by Property Type
```python
# Visualization 5: Boxplot of Median Sale Price by Property Type
plt.figure(figsize=(12, 8))
sns.boxplot(data=Redfin_df_cleaned, x='property_type', y='median_sale_price', palette="Set3",legend=False, hue = 'property_type')
plt.title("Median Sale Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Median Sale Price")
plt.xticks(rotation=45)
plt.show()
```
<img width="920" alt="Screenshot 2025-02-24 at 9 07 54 PM" src="https://github.com/user-attachments/assets/c57026cf-8699-41fa-a5b0-d4b3cd50ea35" />

#### Visualization 6: Density Plot of Sale Price by Property Type
```python
# Visualization 6: Density Plot of Sale Price by Property Type
plt.figure(figsize=(12, 8))
sns.kdeplot(data=Redfin_df_cleaned, x='median_sale_price', hue='property_type', fill=True, common_norm=False, alpha=0.6)
plt.title("Density Plot of Median Sale Price by Property Type")
plt.xlabel("Median Sale Price")
plt.ylabel("Density")
plt.show()
```
<img width="932" alt="Screenshot 2025-02-24 at 9 09 36 PM" src="https://github.com/user-attachments/assets/cab63aab-b67a-4d9d-8457-9f8e4be9c57d" />

#### Visualization 7: Violin Plot of Days on Market by State
```python
# Visualization 7: Violin Plot of Days on Market by State
states = Redfin_df_cleaned['state'].value_counts().index
filtered_data = Redfin_df_cleaned[Redfin_df_cleaned['state'].isin(states)]

plt.figure(figsize=(14, 8))
sns.violinplot(data=filtered_data, x='state', y='median_dom', palette="muted", legend=False,hue = 'state',density_norm='width' )
plt.title("Violin Plot of Median Days on Market by States")
plt.xlabel("State")
plt.ylabel("Median Days on Market")
plt.xticks(rotation=45)
plt.show()
```
<img width="1071" alt="Screenshot 2025-02-24 at 9 10 48 PM" src="https://github.com/user-attachments/assets/1819362a-0ddb-4a08-83b0-3e94163b78dc" />

#### Visualization 8: Heatmap of Median Sale Price by State and Property Type
```python
# Visualization 8: Heatmap of Median Sale Price by State and Property Type
pivot_table = Redfin_df_cleaned.pivot_table(
    values='median_sale_price',
    index='state',
    columns='property_type',
    aggfunc='median'
)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='coolwarm', cbar=True)
plt.title("Heatmap of Median Sale Price by State and Property Type")
plt.xlabel("Property Type")
plt.ylabel("State")
plt.xticks(rotation=45)
plt.show()
```
<img width="841" alt="Screenshot 2025-02-24 at 9 12 14 PM" src="https://github.com/user-attachments/assets/8a39c414-3920-4d8f-b660-e3371ee39d98" />

#### Visualization 9: Pair Plot for Key Features
```python
# Visualization 9: Pair Plot for Key Features
key_features = ['median_sale_price', 'median_list_price', 'median_ppsf', 'homes_sold', 'inventory']

sns.pairplot(Redfin_df_cleaned[key_features], diag_kind="kde", plot_kws={"alpha": 0.7})
plt.suptitle("Pair Plot for Key Features", y=1.02)
plt.show()
```
<img width="1006" alt="Screenshot 2025-02-24 at 9 13 58 PM" src="https://github.com/user-attachments/assets/116e51b7-2525-4334-86c4-18bb0e9375db" />

#### Visualization 10: Correlation Heatmap
```python
# Visualization 10: Correlation Heatmap

# Extract only numerical columns for correlation analysis
numerical_columns = Redfin_df_cleaned.select_dtypes(include=['float64', 'int64'])
filtered_numerical_columns = numerical_columns.drop(columns=['table_id', 'region_type_id', 'ZipCode', 'property_type_id', 'parent_metro_region_metro_code'], errors='ignore')

# Compute correlation matrix for numerical columns
numerical_corr_matrix = numerical_columns.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_corr_matrix, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap for Numerical Features")
plt.show()
```
<img width="877" alt="Screenshot 2025-02-24 at 9 16 41 PM" src="https://github.com/user-attachments/assets/eeb26f5c-e78a-42f8-8175-c9a8a8367c51" />

#### Key Insights from the Correlation Heatmap:

**Strong Positive Correlations:**
* median_list_price and median_sale_price: A very high positive correlation indicates that list prices are strong predictors of sale prices.
* median_ppsf (price per square foot) and median_sale_price: Indicates that the price per square foot is a significant factor in determining sale prices.

**Moderate Positive Correlations:**
* pending_sales and median_sale_price: Suggests that higher pending sales are associated with higher sale prices.
* homes_sold and median_sale_price: Indicates a link between market activity (homes sold) and property values.

**Negative Correlations:**
* median_dom (days on market) and median_sale_price: Properties with higher prices tend to spend fewer days on the market, reflecting higher demand.
* inventory and median_sale_price: Suggests that increased inventory might pressure prices downward, likely due to supply-demand dynamics.

**Feature Clustering:**
* Price-related metrics (e.g., median_list_price, median_ppsf) are tightly correlated, forming a cluster of features with shared predictive power.
* Market activity metrics (e.g., homes_sold, pending_sales) show interdependence, highlighting their mutual influence on market conditions.

**Irrelevant or Weak Correlations:**
* Metrics like off_market_in_two_weeks and its derivatives have little correlation with median_sale_price, suggesting limited predictive value for these features.

#### Identify key correlations with `median_sale_price`
```python
# Identify key correlations with `median_sale_price`
key_correlations = numerical_corr_matrix['median_sale_price'].sort_values(ascending=False).iloc[1:6]

# Scatter plots for key correlations
for feature in key_correlations.index:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=numerical_columns, x=feature, y='median_sale_price', alpha=0.6)
    plt.title(f"Scatter Plot: {feature} vs Median Sale Price")
    plt.xlabel(feature)
    plt.ylabel("Median Sale Price")
    plt.grid(True)
    plt.show()
```

#### Identify top features based on correlation with `median_sale_price`
```python
# Identify top features based on correlation with `median_sale_price`
correlations_with_target = numerical_corr_matrix['median_sale_price'].sort_values(ascending=False)

# Select the top 10 features most correlated with `median_sale_price` (excluding itself)
top_features = correlations_with_target.iloc[1:11]

# Display the top features
top_features
```
<img width="422" alt="Screenshot 2025-02-24 at 9 25 14 PM" src="https://github.com/user-attachments/assets/06e2c212-5672-4e8c-96d2-66e3a408b4ef" />

#### Top Features for Modeling:

Based on their correlation with median_sale_price, the top features are:

* median_list_price: (Correlation: 0.90) The strongest predictor of sale prices.
* avg_sale_to_list: Indicates how closely sale prices match listing prices.
* sold_above_list: Reflects competitive markets where properties sell above the asking price.
* median_ppsf: Price per square foot, an essential metric for value estimation.
* median_list_ppsf: Listed price per square foot, a precursor for market valuation.
* off_market_in_two_weeks: Percentage of properties off the market quickly, indicating high demand.
* median_sale_price_mom: Month-over-month change in sale prices, indicating trends.
* median_dom_yoy: Year-over-year change in days on the market, showing shifts in market dynamics.
* homes_sold: A measure of market activity and absorption.
* new_listings: Supply-side indicator for available inventory.

### Models

#### 1. Linear Regression Model

```python
selected_features = ['median_list_price', 'avg_sale_to_list', 'sold_above_list',
                     'median_ppsf', 'median_list_ppsf', 'off_market_in_two_weeks',
                     'median_sale_price_mom', 'median_dom_yoy', 'homes_sold', 'new_listings']
X = Redfin_df_cleaned[selected_features]
y = Redfin_df_cleaned['median_sale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model
rmse_updated = mean_squared_error(y_test, y_pred)** 0.5
r2_updated = r2_score(y_test, y_pred)

rmse_updated, r2_updated
```
#### The linear regression model has been built and evaluated:
* Root Mean Squared Error (RMSE): $169,626.27 (indicates the average prediction error in monetary terms).
* R² Score: 0.725 (72.5% of the variance in house prices is explained by the model).

#### Feature Engineering on the selected Features
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Perform feature engineering on the selected features

# 1. Interaction terms: Create new features combining related columns
X['list_price_ppsf_interaction'] = X['median_list_price'] * X['median_list_ppsf']
X['sale_to_list_interaction'] = X['avg_sale_to_list'] * X['sold_above_list']

# 2. Log transformation: Handle skewed features
X['log_median_list_price'] = np.log1p(X['median_list_price'])
X['log_median_ppsf'] = np.log1p(X['median_ppsf'])

# 3. Standardize the features for uniform scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Re-train the linear regression model with engineered features
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions with the updated model
y_pred = linear_model.predict(X_test)

# Evaluate the improved model
rmse_improved = mean_squared_error(y_test, y_pred) ** 0.5
r2_improved = r2_score(y_test, y_pred)

rmse_improved, r2_improved
```
#### The feature engineering process led to a significant decline in model performance:
* Root Mean Squared Error (RMSE): $1,180,907.16 (higher than before, indicating larger errors).
* R² Score: -12.32 (negative, suggesting the model is failing to explain variance).

#### Analyze feature importance in the model
```python
# Extract and analyze feature importance (coefficients) from the linear regression model
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': linear_model.coef_
}).sort_values(by='Coefficient', ascending=False)

# Display the feature importance using a standard pandas method
# Instead of: import ace_tools as tools; tools.display_dataframe_to_user(name="Feature Importance Analysis", dataframe=feature_importance)
print("Feature Importance Analysis:\n")

feature_importance
```
<img width="333" alt="Screenshot 2025-02-24 at 9 35 03 PM" src="https://github.com/user-attachments/assets/6d94ba80-a5fc-4e8b-b14a-e5994460712e" />

The feature importance analysis for the linear regression model has been completed, ranking features by their coefficients.

Key observations:
* Positive contributors: Features like median_list_ppsf, median_list_price, and interaction terms (sale_to_list_interaction) have strong positive impacts on median_sale_price.
* Negative contributors: Features like list_price_ppsf_interaction and sold_above_list negatively influence the target variable.

#### Model 2: Logistic Regression:
```python
# prompt: Run Logistic Regression on the Redfin_df_cleaned

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scaling


# Assuming 'median_sale_price' is your target variable for classification
# You might need to categorize 'median_sale_price' into different classes
# (e.g., low, medium, high) to apply logistic regression
# Example categorization:
Redfin_df_cleaned['price_category'] = pd.cut(Redfin_df_cleaned['median_sale_price'], bins=3, labels=['Low', 'Medium', 'High']) # This line was commented out, causing the error. Uncomment to create the 'price_category' column.


# Select features and target
# Assuming you already selected relevant features
selected_features = ['median_list_price', 'avg_sale_to_list', 'sold_above_list',
                     'median_ppsf', 'median_list_ppsf', 'off_market_in_two_weeks',
                     'median_sale_price_mom', 'median_dom_yoy', 'homes_sold', 'new_listings']
X = Redfin_df_cleaned[selected_features]

# Target variable needs to be categorical for Logistic Regression
y = Redfin_df_cleaned['price_category'] #replace with your categorical target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize and train the model
# Increased max_iter further and added a different solver
model = LogisticRegression(max_iter=5000, solver='saga')  
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```
<img width="480" alt="Screenshot 2025-02-24 at 9 51 07 PM" src="https://github.com/user-attachments/assets/351f6ca4-6759-41aa-b504-6df124e35856" />

#### Logistic Model Analysis
Overall Accuracy: 90.5%

Precision, Recall, and F1-score:
* High: Precision = 0.83, Recall = 0.74, F1-score = 0.78
* Low: Precision = 0.95, Recall = 0.95, F1-score = 0.95
* Medium: Precision = 0.85, Recall = 0.87, F1-score = 0.86

The model performs well in classifying Low and Medium price categories but has slightly lower recall for High-priced properties, indicating some misclassification in this segment.


#### Model 3: Ridge/Lasso Regression
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

# Define features and target variable
X = Redfin_df_cleaned[selected_features]
y = Redfin_df_cleaned['median_sale_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge and Lasso models
ridge_model = Ridge(alpha=1.0, random_state=42)
lasso_model = Lasso(alpha=0.1, random_state=42)

# Train Ridge and Lasso models
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Evaluate Ridge model
y_pred_ridge = ridge_model.predict(X_test)
rmse_ridge = mean_squared_error(y_test, y_pred_ridge) ** 0.5
r2_ridge = r2_score(y_test, y_pred_ridge)

# Evaluate Lasso model
y_pred_lasso = lasso_model.predict(X_test)
rmse_lasso = mean_squared_error(y_test, y_pred_lasso) ** 0.5
r2_lasso = r2_score(y_test, y_pred_lasso)

# Display results
ridge_lasso_comparison = pd.DataFrame({
    'Model': ['Ridge Regression', 'Lasso Regression'],
    'RMSE': [rmse_ridge, rmse_lasso],
    'R²': [r2_ridge, r2_lasso]
})


print("Ridge vs Lasso Regression Comparison:\n")

ridge_lasso_comparison
```
<img width="348" alt="Screenshot 2025-02-24 at 9 57 17 PM" src="https://github.com/user-attachments/assets/8ebd7564-2080-4514-a0f6-12a3b57575a9" />

#### Extract feature coefficients from Ridge and Lasso models
```python
# Extract feature coefficients from Ridge and Lasso models
ridge_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Ridge Coefficient': ridge_model.coef_
}).sort_values(by='Ridge Coefficient', ascending=False)

lasso_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Lasso Coefficient': lasso_model.coef_
}).sort_values(by='Lasso Coefficient', ascending=False)

# Merge and compare Ridge and Lasso coefficients
coefficients_comparison = pd.merge(ridge_coefficients, lasso_coefficients, on='Feature')

# Instead of: import ace_tools as tools; tools.display_dataframe_to_user(name="Ridge vs Lasso Coefficients", dataframe=coefficients_comparison)
print("Ridge vs Lasso Coefficients:\n")

# Display the DataFrame using a standard pandas method

coefficients_comparison
```
<img width="516" alt="Screenshot 2025-02-24 at 10 00 24 PM" src="https://github.com/user-attachments/assets/ece8c81d-21d4-4f28-9c08-8923a48e592a" />

#### Key Insights:
avg_sale_to_list:
* Ridge: $463,478
* Lasso: $477,916

Top Features:

* median_sale_price_mom: Significant in both models, indicating the importance of recent price trends.

Moderate Contributors:

* sold_above_list: Consistently contributes positively across both models, reflecting the competitiveness of markets.
* off_market_in_two_weeks: A smaller but positive influence.

Negatively Contributing Features:

* new_listings: Negatively impacts predictions, likely due to supply-side pressure on prices.
* median_list_ppsf: Negligible negative coefficients, indicating minimal contribution to prediction.

Differences:

Lasso Regression applies stronger regularization, leading to slightly different coefficients (e.g., larger for avg_sale_to_list and sold_above_list).

#### Both Models Residuels Analysis
```python
# Calculate residuals for both models
Redfin_df_cleaned['ridge_residuals'] = Redfin_df_cleaned['median_sale_price'] - ridge_model.predict(X)
Redfin_df_cleaned['lasso_residuals'] = Redfin_df_cleaned['median_sale_price'] - lasso_model.predict(X)

# Plot residuals for Ridge Regression
plt.figure(figsize=(12, 6))
plt.scatter(Redfin_df_cleaned['median_sale_price'], Redfin_df_cleaned['ridge_residuals'], alpha=0.5, label='Ridge Residuals')
plt.axhline(0, color='red', linestyle='--', label='Ideal Residuals (0)')
plt.title("Residuals for Ridge Regression")
plt.xlabel("Actual Sale Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals for Lasso Regression
plt.figure(figsize=(12, 6))
plt.scatter(Redfin_df_cleaned['median_sale_price'], Redfin_df_cleaned['lasso_residuals'], alpha=0.5, label='Lasso Residuals', color='orange')
plt.axhline(0, color='red', linestyle='--', label='Ideal Residuals (0)')
plt.title("Residuals for Lasso Regression")
plt.xlabel("Actual Sale Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.grid(True)
plt.show()
```
<img width="469" alt="Screenshot 2025-02-24 at 10 05 38 PM" src="https://github.com/user-attachments/assets/e7a00c44-6988-4b0b-9105-ee0d55775f6e" />

#### Residual Analysis:

**Ridge Regression Residuals:**

* Residuals are scattered relatively evenly around the zero line, indicating that the model performs consistently across different sale price ranges.
* Some larger residuals are visible for extreme sale prices, suggesting areas where the model struggles to predict accurately.

**Lasso Regression Residuals:**

* Residuals are similar to Ridge but show slightly more clustering near zero, indicating slightly better alignment with actual sale prices.
* Outliers and deviations are still present for extreme values, similar to Ridge Regression.
