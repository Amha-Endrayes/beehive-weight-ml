import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:
        pass
    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)

class BeehiveFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Thermodynamic and biological feature engineering transformer:
    - Brood core temperature (t_i_3) vs peripheral clearance (t_i_1, t_i_5)
    - Thermal heterogeneity (standard deviation across internal probes)
    - Sensible heat flux driving force (internal mean - outside temperature)
    - Continuous cyclical temporal representations (sin/cos of hour and day of year)
    - Daylight solar elevation proxy
    - Dynamic thermal rate of change and historical lags
    """
    def __init__(self, include_lags=True, lag_steps=(1, 3, 6, 24)):
        self.include_lags = include_lags
        self.lag_steps = lag_steps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'], utc=True)
            df['hour'] = df['dt'].dt.hour + df['dt'].dt.minute / 60.0
            df['doy'] = df['dt'].dt.dayofyear
        
        # Internal sensor probe aggregations
        temp_cols = [c for c in ['t_i_1', 't_i_2', 't_i_3', 't_i_4', 't_i_5'] if c in df.columns]
        if temp_cols:
            df['temp_internal_mean'] = df[temp_cols].mean(axis=1)
            df['temp_internal_std'] = df[temp_cols].std(axis=1).fillna(0.0)
            df['temp_internal_min'] = df[temp_cols].min(axis=1)
            df['temp_internal_max'] = df[temp_cols].max(axis=1)
            
            if 't_i_3' in df.columns and 't_i_1' in df.columns and 't_i_5' in df.columns:
                df['temp_core_periph_grad'] = df['t_i_3'] - 0.5 * (df['t_i_1'] + df['t_i_5'])
        
        # Ambient-internal thermal gradients
        if 'temp_internal_mean' in df.columns and 't_o' in df.columns:
            df['temp_ambient_grad'] = df['temp_internal_mean'] - df['t_o']
        elif 't' in df.columns and 't_o' in df.columns:
            df['temp_ambient_grad'] = df['t'] - df['t_o']
            
        # Cyclical continuous temporal encodings
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2.0 * np.pi * df['hour'] / 24.0)
            df['hour_cos'] = np.cos(2.0 * np.pi * df['hour'] / 24.0)
        if 'doy' in df.columns:
            df['doy_sin'] = np.sin(2.0 * np.pi * df['doy'] / 365.25)
            df['doy_cos'] = np.cos(2.0 * np.pi * df['doy'] / 365.25)
            
        # Solar proxy
        if 'hour' in df.columns and 'doy' in df.columns:
            solar_proxy = np.maximum(0.0, np.cos(2 * np.pi * (df['hour'] - 13.0) / 24.0)) * \
                          (0.5 + 0.5 * np.cos(2 * np.pi * (df['doy'] - 172) / 365.25))
            df['solar_proxy'] = solar_proxy

        # Historical lags if colony key present
        if self.include_lags and 'key' in df.columns and 'dt' in df.columns:
            df = df.sort_values(['key', 'dt']).reset_index(drop=True)
            for lag in self.lag_steps:
                if 't_o' in df.columns:
                    df[f't_o_lag_{lag}'] = df.groupby('key')['t_o'].shift(lag)
                if 't_i_3' in df.columns:
                    df[f't_core_lag_{lag}'] = df.groupby('key')['t_i_3'].shift(lag)
            if 't_o' in df.columns and 't_o_lag_1' in df.columns:
                df['dt_o_1h'] = df['t_o'] - df['t_o_lag_1']

        # Exclude metadata columns
        drop_cols = ['timestamp', 'dt', 'weight_kg', 'weight_kg_noOutlier', 
                     'weight_delta_noOutlier', 'key', 'colony_id']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        return df[feature_cols].fillna(0.0)
