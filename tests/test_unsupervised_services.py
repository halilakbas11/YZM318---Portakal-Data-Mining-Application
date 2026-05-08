from __future__ import annotations

import numpy as np
import polars as pl

from portakal_app.data.services.distance_file_service import DistanceFileService
from portakal_app.data.services.distance_transformation_service import DistanceTransformationService
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.outliers_service import OutliersService


def test_distance_file_service_loads_labeled_csv(tmp_path):
    path = tmp_path / "distances.csv"
    path.write_text(",A,B,C\nA,0,1,2\nB,1,0,3\nC,2,3,0\n", encoding="utf-8")

    handle = DistanceFileService().load(str(path))

    assert handle.matrix.shape == (3, 3)
    assert handle.row_labels == ("A", "B", "C")
    assert float(handle.matrix[1, 2]) == 3.0


def test_distance_transformation_service_transforms_matrix():
    service = DistanceTransformationService()
    handle = service.transform(
        [[0.0, 2.0], [2.0, 0.0]],
        normalization="zero_one",
        inversion="one_minus",
    )

    assert handle.matrix.shape == (2, 2)
    assert np.isclose(handle.matrix[0, 1], 0.0)
    assert np.isclose(handle.matrix[0, 0], 0.0)


def test_outliers_service_splits_outliers_and_inliers():
    frame = pl.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 8.0],
            "y": [0.0, 0.2, 0.1, 8.5],
            "label": ["A", "A", "A", "B"],
        }
    )
    dataset = GeneratedDatasetService().build_dataset(
        frame,
        dataset_id="outlier-test",
        display_name="Outlier Test",
        file_name="outlier-test.csv",
    )

    outputs = OutliersService().detect(
        dataset,
        method="isolation_forest",
        contamination=0.25,
        neighbors=2,
        nu=0.25,
        gamma="scale",
        support_fraction=1.0,
    )

    assert outputs["Data"].row_count == 4
    assert outputs["Outliers"].row_count >= 1
    assert outputs["Inliers"].row_count + outputs["Outliers"].row_count == 4
    assert "Outlier" in outputs["Data"].dataframe.columns
