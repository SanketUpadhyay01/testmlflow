import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

experiment_name = 'California_Price_Comparison'
try:
    exp_id = mlflow.create_experiment(name=experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

def log_model_and_metrics(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param('model_name', model_name)
        if hasattr(model, 'n_estimators'):
            mlflow.log_param('n_estimators', model.n_estimators)
        if hasattr(model, 'learning_rate'):
            mlflow.log_param('learning_rate', model.learning_rate)
        if hasattr(model, 'max_depth'):
            mlflow.log_param('max_depth', model.max_depth)

        mlflow.log_metric('mae', mae)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)

        mlflow.sklearn.log_model(model, model_name)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name} Predictions vs Actual')
        plt.grid()

        plot_filename = f"{model_name}_predictions.png"
        plt.savefig(plot_filename)
        plt.close()
        mlflow.log_artifact(plot_filename)
    
    return mae, mse, r2

# Linear Regression Model
lr_model = LinearRegression().fit(X_train, y_train)
log_model_and_metrics(lr_model, 'Linear_Regression', X_test, y_test)

# AdaBoost Regressor model
abr_model = AdaBoostRegressor(n_estimators=100, learning_rate=1.3, random_state=42).fit(X_train, y_train)
log_model_and_metrics(abr_model, 'AdaBoost_Regressor', X_test, y_test)

# Decision Tree Regressor model
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
log_model_and_metrics(dt_model, 'Decision_Tree_Regressor', X_test, y_test)

# Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
log_model_and_metrics(rf_model, 'Random_Forest_Regressor', X_test, y_test)

mlflow.set_tracking_uri("https://localhost:7501")
