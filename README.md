# Predicting California housing prices

This is a Machine Learning based project to predict housing prices in Califorina based on some features which are ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]. By applying regression Models like ( Random Forest and XGB ). We achieved more than 0.80 in R^2 test.


## Requirements

- Python 3.12 or later


## Install the required packages

```bash
$ pip install -r requirements.txt
```


## Dataset Description

- The Dataset is separated to california_housing_train.csv [ which contains 17,000 rows with 9 Columns ] and california_housing_test.csv [ which contains 3,000 rows with 9 Columns ].



## Baseline Model

- After performing EDA, we noticed that most features had very weak linear correlation with the target variable, except for `median_income` (correlation = 0.67). 
This suggested that the dataset contains **non-linear relationships**, which linear models cannot capture well. 

Therefore, we moved to tree-based models such as **Random Forest** and **XGBoost**, which are more suitable for handling non-linearity.

## Grid Search

- We used RandomizedSearchCV (Scikit-Learn library) on RandomForestRegressor model and got the best parameters {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 0.5, 'max_depth': None}

- We used RandomizedSearchCV (Scikit-Learn library) on XGB model (xgboost library) and got the best parameters {'subsample': 0.6, 'reg_lambda': 5, 'reg_alpha': 0.1, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 1.0}

## Cross-Validation

- We used **k-flod cross-validation** to ensure that the model generalizes well to unseen data and to reduce the risk of overfitting.


## Evaluation

|           | Random Forest   | XGB      |
| :-------  | :------:        | -------: |
| CV RMSE   | 76,525.54       | 71,658.83|
| Test RMSE | 50,391.95       | 49,936.83|
| Test MAE  | 33,106.83       | 33,430.40|
| Test R2   | 0.801           | 0.805    |


## Conclusion
After seeing the Results we can say **XGB** model is slightly better than **Random Forest Regressor** model. In the Future we will try using **LightGBM**.
