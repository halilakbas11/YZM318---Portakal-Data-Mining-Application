import polars as pl
from portakal_app.data.services.continuize_service import ContinuizeService
from portakal_app.data.models import DatasetHandle, build_data_domain
import json

df = pl.DataFrame({
    "A": ["a", "b", "c", "a", "c", "b"]
})
domain = build_data_domain(df)
dataset = DatasetHandle(dataset_id="1", display_name="Test", dataframe=df, row_count=6, column_count=1, domain=domain)

service = ContinuizeService()

res1 = service.continuize(dataset, discrete_preset="Treat as ordinal")
print("Treat as ordinal:", res1.dataframe.columns)

res2 = service.continuize(dataset, discrete_preset="One-hot encoding")
print("One-hot encoding:", res2.dataframe.columns)

res3 = service.continuize(dataset, discrete_preset="First value as base")
print("First value as base:", res3.dataframe.columns)
