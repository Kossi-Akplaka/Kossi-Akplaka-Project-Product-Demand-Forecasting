# Product Demand Forecasting: Project Overview
* Utilized Python and time series techniques to forecast product demand
* Performed exploratory data analysis by exploring the features and the relationship between those features
* Engineered features and used hyperparameter tunning to reach the best model 

## Data and packages
* Data: https://www.kaggle.com/datasets/felixzhao/productdemandforecasting
* Packages: Pandas, Numpy, Matplotlib, Seaborn, Pickle

## Initial Features
* Product Code
* Warehouse
* Product Category
* Order Date
* Order Demand

# Univariate analysis
### I looked at the distributions of the data for various categorical and continuous variables. Below are a few highlights.

  ![Warehouse Count](Warehouse_count.png)
  ![Category Count](Category_count.png)

  
  
# Bivariate/ Multivariate Analysis
 ![Demand per Warehouse](Demand_per_warehouse.png)
 ![Average Demand per Warehouse](Average_demand_per_warehouse.png)
 
 
 # Model Building
 ### Created new features 'day_of_the_week', 'Quarter',' Month', 'Year', and 'Week' from the date columns
 ### Split the data between features and target variable 'Demand'
 ### Tried 3 different models:
 ## XGboost:  Sequentially built shallow decision trees to provide accurate results and a highly scalable training method that avoids overfitting.
 ## Arima: Capture trends in the data and forecasting 
 ## Sarima: Capture the seasonality in the data
 
 # Model performance: the SARIMA model outperformed the other models on the test and validation sets
 ### XGboost:  RMSE = 30754.97
 ### Arima: RMSE = 387.30
 ### Sarima: RMSE = 309.61
 
 # Saved the model with Pickle for future use
 
      
