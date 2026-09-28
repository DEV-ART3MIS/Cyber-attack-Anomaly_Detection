from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def train_model(X, y, model_type="linear", degree=3):
    
    if model_type == "polynomial":
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        return model, poly
    
    else:
        model = LinearRegression()
        model.fit(X, y)
        return model, None


def predict(model, X, poly=None):
    if poly:
        X = poly.transform(X)
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"r2": r2, "mae": mae, "mse": mse}
