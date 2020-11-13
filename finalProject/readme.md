# Task
- In this project, the objective is to perform binary classification on the loan dataset from the U.S. SBA which includes historical data from 1987 to 2014. The task is to predict whether a loan will be charged-off or paid in-full, based on several attributes such as City, ApprovalFY, CreatedJob, GrossApproved, DisbursementAmount and so on. 


# Functions for Different Files
- **Vis.ipynb** is to visualize data. It is part of Exploratory and Data Analysis. 
- **bestFeatureBefore.py** includes all features we made from feature engineering.
- **bestFeatureBeforeCondense.py** includes all features we made except one-hot encoding for _NAICS_
- **simple.py** only consists of all original features without extra feature engineering
- **fullsearch.py** includes grid search or random search for hyperparameters tuning
- **featureImportance.py** can draw feature importance.

