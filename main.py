import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("house_price_data.csv")

# Normalize column names to lowercase
data.columns = data.columns.str.lower()

print("Dataset Preview:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Handle categorical data with one-hot encoding
if data.select_dtypes(include='object').shape[1] > 0:
    data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target variable (y)
if 'price' not in data.columns:
    print("\nError: The dataset does not contain a 'price' column.")
else:
    X = data.drop(columns=['price'])
    y = data['price']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    # Evaluate models
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[model_name] = {"MAE": mae, "MSE": mse, "R2": r2}

    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MAE={metrics['MAE']:.2f}, MSE={
              metrics['MSE']:.2f}, R2={metrics['R2']:.2f}")

    # Hyperparameter tuning for Random Forest
    rf = RandomForestRegressor()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters for Random Forest:")
    print(grid_search.best_params_)
    print(f"Best R2 Score: {grid_search.best_score_:.2f}")

    # Evaluate the best model on test data
    best_model = grid_search.best_estimator_
    final_predictions = best_model.predict(X_test)

    print("\nFinal Model Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, final_predictions):.2f}")
    print(f"MSE: {mean_squared_error(y_test, final_predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, final_predictions):.2f}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, final_predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(),
             y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.show()
