from sklearn.ensemble import RandomForestRegressor

def get_random_forest():
    return RandomForestRegressor(random_state=0)

param_grid_rf = {
    "regressor__n_estimators": [100, 500],
    "regressor__max_depth": [20, 50],   
}