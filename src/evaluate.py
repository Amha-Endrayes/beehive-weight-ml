import numpy as np

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    def mean_squared_error(y_true, y_pred):
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    def r2_score(y_true, y_pred):
        yt, yp = np.array(y_true), np.array(y_pred)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-9)))

def calculate_metrics(y_true, y_pred):
    """
    Returns standard error metrics:
    - MAE (g)
    - RMSE (g)
    - Median Absolute Error (g)
    - R-squared (R2)
    """
    mae_g = mean_absolute_error(y_true, y_pred) * 1000.0
    rmse_g = np.sqrt(mean_squared_error(y_true, y_pred)) * 1000.0
    r2 = r2_score(y_true, y_pred)
    med_ae_g = np.median(np.abs(y_true - y_pred)) * 1000.0
    return {
        'MAE (g)': round(mae_g, 1),
        'RMSE (g)': round(rmse_g, 1),
        'MedianAE (g)': round(med_ae_g, 1),
        'R2': round(r2, 4)
    }
