import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

thresholds = {
    "mae":0.8,
    "mse":0.8,
    "r2":0.4
}

def train_and_evaluate(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def test_linear_regression():
    model = LinearRegression()
    mae, mse, r2 = train_and_evaluate(model)
    assert mae <= thresholds['mae'], f"Linear Regression MAE {mae} is greater than threshold"
    assert mse <= thresholds['mse'], f"Linear Regression MSE {mse} is greater than threshold"
    assert r2 >= thresholds['r2'], f"Linear Regression R2 {r2} is less than threshold"

def test_adaboost_regression():
    model= AdaBoostRegressor(n_estimators=100, random_state=42)
    mae, mse, r2 = train_and_evaluate(model)
    assert mae <= thresholds['mae'], f"AdaBoost Regression MAE {mae} is greater than threshold"
    assert mse <= thresholds['mse'], f"AdaBoost Regression MSE {mse} is greater than threshold"
    assert r2 >= thresholds['r2'], f"AdaBoost Regression R2 {r2} is less than threshold"

def test_decisiontree_regression():
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    mae, mse, r2 = train_and_evaluate(model)
    assert mae <= thresholds['mae'], f"Decision Tree Regression MAE {mae} is greater than threshold"
    assert mse <= thresholds['mse'], f"Decision Tree Regression MSE {mse} is greater than threshold"
    assert r2 >= thresholds['r2'], f"Decision Tree Regression R2 {r2} is less than threshold"

def test_randomforest_regression():
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    mae, mse, r2 = train_and_evaluate(model)
    assert mae <= thresholds['mae'], f"Random Forest Regression MAE {mae} is greater than threshold"
    assert mse <= thresholds['mse'], f"Random Forest Regression MSE {mse} is greater than threshold"
    assert r2 >= thresholds['r2'], f"Random Forest Regression R2 {r2} is greater than threshold"
