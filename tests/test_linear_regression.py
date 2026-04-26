import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from portakal_app.data.models import DatasetHandle, DataDomain, ColumnSchema
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
import polars as pl

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.svc = SklearnLearnerService()
        
    def test_linear_regression_fit(self):
        # Create a simple numeric dataset
        df = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [0.0, 0.0, 0.0, 0.0, 0.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        domain = DataDomain(
            columns=(
                ColumnSchema("x1", "f64", "numeric", "feature", False, 0, 0),
                ColumnSchema("x2", "f64", "numeric", "feature", False, 0, 0),
                ColumnSchema("y", "f64", "numeric", "target", False, 0, 0)
            )
        )
        dataset = DatasetHandle(
            dataset_id="test-ds",
            display_name="Test Dataset",
            source=None,  # Not used by fit
            domain=domain,
            dataframe=df,
            row_count=5,
            column_count=3,
            cache_path=None
        )
        
        est = LinearRegression()
        result = self.svc.fit(est, dataset, "Linear Regression", "linear_regression")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.display_name, "Linear Regression")
        self.assertTrue(hasattr(result.trained_model, "coef_"))
        
        # y = 2 * x1 + 0 * x2
        coefs = result.trained_model.coef_
        self.assertAlmostEqual(coefs[0], 2.0, places=5)
        self.assertAlmostEqual(coefs[1], 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
