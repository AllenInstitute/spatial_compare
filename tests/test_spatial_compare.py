from spatial_compare import SpatialCompare, get_column_ordering
from spatial_compare.utils import (
    create_test_data_from_spatial,
    spatial_detection_scores,
)
import pandas as pd
import anndata as ad
import numpy as np
import pathlib
from scipy import sparse

SC_DIR = pathlib.Path(__file__).resolve().parents[1]
print(SC_DIR)

DATA_STEMS = [
    "CJ_BG_mini1.h5ad",
    "CJ_BG_mini2.h5ad",
    "CJ_BG_mini3.h5ad",
    "CJ_BG_mini4.h5ad",
]

TEST_DIR = pathlib.Path(__file__).resolve().parent.joinpath("data")

TEST_ANNDATAS = [ad.read_h5ad(TEST_DIR.joinpath(DATA_STEM)) for DATA_STEM in DATA_STEMS]

TEST_DF_RECORDS = [
    dict(a=1, b=0.5, c=0.1),
    dict(a=0.5, b=0.8, c=0.9),
    dict(a=0.1, b=1.0, c=0.1),
]
TEST_DF = pd.DataFrame.from_records(TEST_DF_RECORDS)
TEST_DF.index = ["a", "b", "c"]


def test_get_column_ordering():
    ordered_columns = get_column_ordering(TEST_DF, ordered_rows=["a", "b", "c"])
    print(ordered_columns)
    assert ordered_columns == ["a", "c", "b"]


def test_SpatialCompare():
    # mock up test
    sc = SpatialCompare(TEST_ANNDATAS[0], TEST_ANNDATAS[1])
    assert all(sc.ad_0[0].obs.columns == sc.ad_1[1].obs.columns)


def _strong_loss_data():
    np.random.seed(42)
    return create_test_data_from_spatial(
        TEST_ANNDATAS[0],
        "supercluster_name",
        spatial_coords=["x_centroid", "y_centroid"],
        shape="square",
        size=0.4,
        location="lower-right",
        loss_factor=0.5,
        noise_factor=0.1,
    )


def _lower_right_mask(adata):
    coordinates = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
    normalized_coordinates = (coordinates - coordinates.min(axis=0)) / np.ptp(
        coordinates, axis=0
    )
    return np.all(
        np.abs(normalized_coordinates - np.array([0.8, 0.2])) <= 0.2,
        axis=1,
    )


def test_create_test_data_from_spatial_strong_loss():
    ad1 = TEST_ANNDATAS[0]
    strong_loss_data = _strong_loss_data()
    affected = _lower_right_mask(ad1)

    assert affected.any()
    assert (~affected).any()
    assert sparse.issparse(strong_loss_data.X)
    assert (strong_loss_data.X[~affected] != ad1.X[~affected]).nnz == 0

    observed_loss_factor = strong_loss_data.X[affected].sum() / ad1.X[affected].sum()
    assert np.isclose(observed_loss_factor, 0.5, atol=0.01)


def test_spatial_detection_scores_detect_strong_loss():
    ad1 = TEST_ANNDATAS[0]
    strong_loss_data = _strong_loss_data()
    affected = _lower_right_mask(ad1)
    strong_loss_data.obs["transcript_counts"] = np.asarray(
        strong_loss_data.X.sum(axis=1)
    ).ravel()

    reference_stats = (
        ad1.obs.groupby("supercluster_name", observed=True)["transcript_counts"]
        .agg(["mean", "std"])
        .dropna()
    )
    results = spatial_detection_scores(
        strong_loss_data.obs,
        reference_stats["mean"],
        reference_stats["std"],
        plot_stuff=False,
        n_bins=20,
    )

    z_scores = results["detection_scores"]["detection_relative_z_score"]
    affected_index = strong_loss_data.obs_names[affected].intersection(z_scores.index)
    unaffected_index = strong_loss_data.obs_names[~affected].intersection(
        z_scores.index
    )

    affected_mean = z_scores.loc[affected_index].mean()
    unaffected_mean = z_scores.loc[unaffected_index].mean()
    assert affected_mean < -0.5
    assert affected_mean < unaffected_mean - 0.5
    assert results["z_score_image"].shape == (20, 20)
