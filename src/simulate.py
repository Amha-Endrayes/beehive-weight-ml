import numpy as np
import pandas as pd

def generate_realistic_beehive_data(num_colonies=6, days=60, freq='1h', seed=42):
    """
    Biologically and thermodynamically realistic honeybee telemetry generator
    matching the empirical distributions of the German honeybee dataset (Zenodo 10407693).
    """
    np.random.seed(seed)
    start_date = pd.Timestamp('2022-04-01', tz='UTC')
    date_range = pd.date_range(start=start_date, periods=days * 24, freq=freq)
    records = []
    
    colony_keys = [f"colony_{i:02d}" for i in range(1, num_colonies + 1)]
    
    for c_idx, key in enumerate(colony_keys):
        box_tare = np.random.uniform(24.0, 46.0)
        colony_strength = np.random.uniform(0.8, 1.2)
        seasonal_nectar = 0.0
        lat = 50.9 + np.random.normal(0, 1.0)
        lon = 7.0 + np.random.normal(0, 1.5)
        
        for t in date_range:
            hour = t.hour
            doy = t.dayofyear
            seasonal_base = 8.0 + 13.0 * np.sin(2 * np.pi * (doy - 105) / 365.25)
            diurnal_ambient = 5.5 * np.cos(2 * np.pi * (hour - 14) / 24.0)
            t_o = seasonal_base + diurnal_ambient + np.random.normal(0, 1.2)
            
            p = 1013.25 + 7.0 * np.sin(2 * np.pi * doy / 14.0) + np.random.normal(0, 1.5)
            h = np.clip(72.0 - 16.0 * np.cos(2 * np.pi * (hour - 14) / 24.0) + np.random.normal(0, 3.5), 30.0, 98.0)
            
            # Core brood thermoregulation
            t_i_3 = 34.8 + 0.25 * np.sin(2 * np.pi * hour / 24.0) + np.random.normal(0, 0.12)
            t_i_2 = t_i_3 - 1.1 + 0.1 * (t_o - 20.0) + np.random.normal(0, 0.15)
            t_i_4 = t_i_3 - 0.9 + 0.1 * (t_o - 20.0) + np.random.normal(0, 0.15)
            t_i_1 = t_i_3 - 4.2 + 0.35 * (t_o - 15.0) + np.random.normal(0, 0.25)
            t_i_5 = t_i_3 - 4.0 + 0.35 * (t_o - 15.0) + np.random.normal(0, 0.25)
            t_bme = 0.5 * (t_i_2 + t_i_4) + np.random.normal(0, 0.1)
            
            # Diurnal foraging
            is_day = 6 <= hour <= 20
            foragers_out = 0.0
            if is_day and t_o > 12.0:
                solar_elevation = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 14.0))
                foragers_out = 1.3 * colony_strength * solar_elevation * min(1.0, (t_o - 12.0) / 8.0)
            
            daily_nectar_gain = 0.0
            if hour == 18 and t_o > 16.0 and is_day:
                daily_nectar_gain = 0.7 * colony_strength * np.random.uniform(0.3, 1.2)
            
            evaporation = 0.02 * (not is_day)
            seasonal_nectar = max(0.0, seasonal_nectar + daily_nectar_gain - evaporation)
            
            weight_kg = box_tare + seasonal_nectar - foragers_out + np.random.normal(0, 0.012)
            weight_kg_noOutlier = weight_kg
            
            records.append({
                'timestamp': str(t),
                'key': key,
                't_i_1': round(t_i_1, 2),
                't_i_2': round(t_i_2, 2),
                't_i_3': round(t_i_3, 2),
                't_i_4': round(t_i_4, 2),
                't_i_5': round(t_i_5, 2),
                't': round(t_bme, 2),
                't_o': round(t_o, 2),
                'h': round(h, 2),
                'p': round(p, 2),
                'lat': round(lat, 1),
                'lon': round(lon, 1),
                'weight_kg': round(weight_kg, 3),
                'weight_kg_noOutlier': round(weight_kg_noOutlier, 3)
            })
            
    return pd.DataFrame(records)
