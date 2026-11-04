import unittest
import numpy as np
import pandas as pd
from src.simulate import generate_realistic_beehive_data
from src.features import BeehiveFeatureTransformer
from src.models import CalibratedBeehiveEstimator
from src.evaluate import calculate_metrics

class TestBeehiveMLPipeline(unittest.TestCase):
    def test_data_generation(self):
        df = generate_realistic_beehive_data(num_colonies=2, days=5)
        self.assertEqual(df.shape[0], 2 * 5 * 24)
        self.assertIn('weight_kg', df.columns)
        self.assertIn('t_i_3', df.columns)
        self.assertGreater(df['t_i_3'].mean(), 33.0)

    def test_feature_transformer(self):
        df = generate_realistic_beehive_data(num_colonies=2, days=5)
        transformer = BeehiveFeatureTransformer()
        df_feat = transformer.transform(df)
        self.assertIn('temp_internal_mean', df_feat.columns)
        self.assertIn('temp_core_periph_grad', df_feat.columns)
        self.assertIn('hour_sin', df_feat.columns)
        self.assertIn('hour_cos', df_feat.columns)
        self.assertNotIn('weight_kg', df_feat.columns)

    def test_calibrated_estimator(self):
        df = generate_realistic_beehive_data(num_colonies=3, days=10)
        transformer = BeehiveFeatureTransformer()
        df_feat = transformer.transform(df)
        
        feature_cols = [c for c in df_feat.columns if c not in ['weight_kg', 'key', 'timestamp']]
        estimator = CalibratedBeehiveEstimator()
        estimator.fit(df_feat[feature_cols], df['weight_kg'], colony_ids=df['key'])
        
        preds = estimator.predict(df_feat[feature_cols], calibration_tare=35.0)
        self.assertEqual(len(preds), len(df))
        metrics = calculate_metrics(df['weight_kg'].iloc[:100], preds[:100])
        self.assertIn('MAE (g)', metrics)
        self.assertIn('R2', metrics)

if __name__ == '__main__':
    unittest.main()
