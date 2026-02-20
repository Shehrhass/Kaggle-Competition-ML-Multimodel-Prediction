# Kaggle Assignment # 1 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error)
from sklearn.model_selection import ParameterGrid, GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import sweetviz as sv

train_url = "https://raw.githubusercontent.com/Shehrhass/Kaggle-Competition-ML-Multimodel-Prediction/main/train.csv"
test_url = "https://raw.githubusercontent.com/Shehrhass/Kaggle-Competition-ML-Multimodel-Prediction/main/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

#EDA
eda_report = sv.compare([train_df, "Training Data"], [test_df, "Test Data"], target_feat="SalePrice")
eda_report.show_html("Ames_Housing_EDA_Report.html")

#checking and prepping data in the df before split 
test_type_counts = test_df.dtypes.value_counts()
train_type_counts = train_df.dtypes.value_counts()
print(test_type_counts)
print(train_type_counts)

X_full = train_df.drop(columns=['Id', 'SalePrice'])
Y_full = np.log1p(train_df['SalePrice'])

X_full['MSSubClass'] = X_full['MSSubClass'].astype(str) #numeric but var is categorical by definition
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)

#Splitting data
X_train, X_val, Y_train, Y_val = train_test_split(X_full, Y_full, test_size=0.2, random_state=42)

# Accounting for null/missing/NaN
Mode_columns = ['GarageCars', 'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath'] #these can't be in float and are always integers thus mode is used
Mean_columns = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1' , 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','GarageArea'] #these can be float thus mean is used

train_type_counts = train_df.dtypes.value_counts()
test_type_counts = test_df.dtypes.value_counts()
print(train_type_counts)
print(test_type_counts)

num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_full.select_dtypes(include=['object']).columns.tolist()

Other_num_cols = [col for col in num_cols if col not in Mode_columns and col not in Mean_columns]

mode_transformer = SimpleImputer(strategy='most_frequent') 
mean_transformer = SimpleImputer(strategy='mean')
median_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('mode_num', mode_transformer, Mode_columns),
        ('mean_num', mean_transformer, Mean_columns),
        ('median_num', median_transformer, Other_num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# hyperparamter tuning 
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 1000],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [2, 5]
}

print("Starting Hyperparameter Tuning...")
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

Y_pred_log = best_model.predict(X_val)
val_rmse = root_mean_squared_error(Y_val, Y_pred_log)
print(f"Validation Log-RMSE: {val_rmse:.4f}")

# multi-model hyperparamter tuning
models_and_grids = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
        },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'model__n_estimators': [100, 300],
            'model__max_depth': [10, None]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42),
        "params": {
            'model__n_estimators': [100, 1000],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [2, 5]
        }
    }
}

best_overall_rmse = float('inf') 
best_overall_model = None
best_overall_name = ""
model_results = {}

# Creating  pipeline for this specific model
for name, config in models_and_grids.items():
    print(f"Training {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', config["model"])])
    
    # Tune the model
    grid_search = GridSearchCV(pipeline, config["params"], cv=3, scoring='neg_root_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train, Y_train)
    
    # Evaluate on the unseen validation set
    current_best_model = grid_search.best_estimator_
    Y_pred_log = current_best_model.predict(X_val)
    val_rmse = root_mean_squared_error(Y_val, Y_pred_log)
        
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Validation Log-RMSE: {val_rmse:.4f}\n")
        
    model_results[name] = val_rmse
        
    # Check if this model beat the previous best
    if val_rmse < best_overall_rmse:
        best_overall_rmse = val_rmse
        best_overall_model = current_best_model
        best_overall_name = name

print(f"Best Overall: {best_overall_name} with Log-RMSE: {best_overall_rmse:.4f}")


# Plot Actual_v_Pred prices (@validation set)
Y_pred_log_best = best_overall_model.predict(X_val)

Y_val_normal = np.expm1(Y_val).values.flatten()
Y_pred_normal = np.expm1(Y_pred_log_best).flatten()

sort_indices = np.argsort(Y_val_normal)
Y_val_sorted = Y_val_normal[sort_indices]
Y_pred_sorted = Y_pred_normal[sort_indices]

plt.figure(figsize=(10,6))
plt.plot(np.arange(len(Y_val_sorted)), Y_val_sorted, label='Original Actual Prices', color='blue', linewidth=2)
plt.plot(np.arange(len(Y_pred_sorted)), Y_pred_sorted, label='Predicted Prices', color='orange', alpha=0.7)
plt.xlabel('House Index (Sorted by Price)')
plt.ylabel('Sale Price ($)')
plt.title('Validation Set: Actual vs Predicted Prices')
plt.legend()
plt.show()

# Collating everything
print("Training final model on full dataset__")
best_overall_model.fit(X_full, Y_full)

X_test = test_df.drop(columns=['Id'])
final_pred_log = best_overall_model.predict(X_test)
final_pred_normal = np.expm1(final_pred_log)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_pred_normal
})

submission.to_csv('output_file_vFn.csv', index=False)

# Kaggle API Submission
os.system(f'kaggle competitions submit -c house-prices-advanced-regression-techniques -f output_file.csv -m "Best Multi-Model: {best_overall_name}"')
print("Submission made to Kaggle.")